import tensorflow as tf
import numpy as np
from .ops import causal_conv, mu_law_encode


def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        return create_variable(name, shape)

def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


class WaveNetModel(object):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2**8,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=32,
                 global_channels=256,
                 local_channels=128,
                 output_channels=64):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
        '''
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width
        self.global_channels = global_channels
        self.local_channels = local_channels
        self.upsample_rate=40
        self.output_channel=output_channels

        self.sparse_params = dict()
        self.variables = self._create_variables()
        self.var_placehold = dict()
        self.up_op = dict()
        self.c_prase = 0
        self.c_params = 0

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):
            with tf.variable_scope('embeddings'):
                layer = dict()

                layer['input_embedding'] = create_embedding_table(
                    'input_embedding',
                    [self.quantization_channels,
                     self.residual_channels])
                var['embeddings'] = layer
            with tf.variable_scope('upsample'):
                current = dict()
                current['filter'] = create_variable(
                    'upsample_filter',
                    [1,
                     self.upsample_rate,
                     self.output_channel,
                     self.local_channels])
                current['filter1'] = create_variable(
                    'upsample_filter',
                    [1,
                     self.output_channel,
                     self.output_channel])
                var['upsample'] = current
            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gcond_filter'] = create_variable(
                            'filter',
                            [self.global_channels,
                             self.dilation_channels])
                        current['gcond_gate'] = create_variable(
                            'gate',
                            [self.global_channels,
                             self.dilation_channels])
                        current['lcond_filter'] = create_variable(
                            'filter',
                            [1,
                             self.output_channel,
                             self.dilation_channels])
                        current['lcond_gate'] = create_variable(
                            'gate',
                            [1,
                             self.output_channel,
                             self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, self.quantization_channels])
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [self.quantization_channels])
                var['postprocessing'] = current

        return var

    def _var2sparse(self,sess,var):
        npvar = var.eval(session=sess)
        np_index = np.where(npvar != 0.0)
        indices = tf.where(tf.not_equal(var,0.0))
        value = npvar[np_index]
        self.c_prase += np.shape(value)[0]
        self.c_params += np.size(npvar)



        sparse_tensor = tf.SparseTensor(indices=indices.eval(session=sess),values=value,shape=var.get_shape())



        return sparse_tensor



    def _create_sparse_tensor(self):

        for layer_index, dilation in enumerate(self.dilations):
            var = self.variables['dilated_stack'][layer_index]
            self.sparse_params['dilated_stack'+str(layer_index)+'filter0']=tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['filter'][0,:,:]).get_shape())
            self.sparse_params['dilated_stack' + str(layer_index) + 'gate0'] = tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['gate'][0,:,:]).get_shape())
            self.sparse_params['dilated_stack' + str(layer_index) + 'lcond_filter'] = tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['lcond_filter'][0,:,:]).get_shape())
            self.sparse_params['dilated_stack' + str(layer_index) + 'lcond_gate'] = tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['lcond_gate'][0,:,:]).get_shape())
            self.sparse_params['dilated_stack'+str(layer_index)+'filter1']=tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['filter'][1,:,:]).get_shape())
            self.sparse_params['dilated_stack' + str(layer_index) + 'gate1'] = tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['gate'][1,:,:]).get_shape())
            self.sparse_params['dilated_stack' + str(layer_index) + 'dense'] = tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['dense'][0,:,:]).get_shape())
            self.sparse_params['dilated_stack' + str(layer_index) + 'skip'] = tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['skip'][0,:,:]).get_shape())
        var = self.variables['postprocessing']
        self.sparse_params['post1'] = tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['postprocess1'][0,:,:]).get_shape())
        self.sparse_params['post2'] = tf.SparseTensor(indices=[[0,0]],values=[0.0],shape=tf.transpose(var['postprocess2'][0,:,:]).get_shape())

    def cread_placehold(self):
        for layer_index, dilation in enumerate(self.dilations):
            var = self.variables['dilated_stack'][layer_index]
            self.var_placehold['dilated_stack' + str(layer_index) + 'filter'] = tf.placeholder(
                dtype=var['filter'].dtype, shape=var['filter'].get_shape())
            self.var_placehold['dilated_stack' + str(layer_index) + 'gate'] = tf.placeholder(dtype=var['gate'].dtype,
                                                                                             shape=var[
                                                                                                 'gate'].get_shape())
            self.var_placehold['dilated_stack' + str(layer_index) + 'lcond_filter'] = tf.placeholder(
                dtype=var['lcond_filter'].dtype, shape=var['lcond_filter'].get_shape())
            self.var_placehold['dilated_stack' + str(layer_index) + 'lcond_gate'] = tf.placeholder(
                dtype=var['lcond_gate'].dtype, shape=var['lcond_gate'].get_shape())
            self.var_placehold['dilated_stack' + str(layer_index) + 'dense'] = tf.placeholder(dtype=var['dense'].dtype,
                                                                                              shape=var[
                                                                                                  'dense'].get_shape())
            self.var_placehold['dilated_stack' + str(layer_index) + 'skip'] = tf.placeholder(dtype=var['skip'].dtype,
                                                                                             shape=var[
                                                                                                 'skip'].get_shape())
            self.up_op['dilated_stack' + str(layer_index) + 'filter'] = var['filter'].assign(
                self.var_placehold['dilated_stack' + str(layer_index) + 'filter'])
            self.up_op['dilated_stack' + str(layer_index) + 'gate'] = var['gate'].assign(
                self.var_placehold['dilated_stack' + str(layer_index) + 'gate'])
            self.up_op['dilated_stack' + str(layer_index) + 'lcond_filter'] = var['lcond_filter'].assign(
                self.var_placehold['dilated_stack' + str(layer_index) + 'lcond_filter'])
            self.up_op['dilated_stack' + str(layer_index) + 'lcond_gate'] = var['lcond_gate'].assign(
                self.var_placehold['dilated_stack' + str(layer_index) + 'lcond_gate'])
            self.up_op['dilated_stack' + str(layer_index) + 'dense'] = var['dense'].assign(
                self.var_placehold['dilated_stack' + str(layer_index) + 'dense'])
            self.up_op['dilated_stack' + str(layer_index) + 'skip'] = var['skip'].assign(
                self.var_placehold['dilated_stack' + str(layer_index) + 'skip'])

        var = self.variables['postprocessing']
        self.var_placehold['post1'] = tf.placeholder(dtype=var['postprocess1'].dtype,
                                                     shape=var['postprocess1'].get_shape())
        self.var_placehold['post2'] = tf.placeholder(dtype=var['postprocess2'].dtype,
                                                     shape=var['postprocess2'].get_shape())

        self.up_op['post1'] = var['postprocess1'].assign(
            self.var_placehold['post1'])
        self.up_op['post2'] = var['postprocess2'].assign(
            self.var_placehold['post2'])
    def prune(self,sess,var, threshold,name):

        npvar = var.eval(session=sess)

        index = np.where(np.abs(npvar) < threshold)
        npvar[index] = 0.0

        self.c_prase += np.shape(index)[1]

        sess.run(self.up_op[name], {self.var_placehold[name]: npvar})

    def thread_prune(self,sess,threshold):
        self.c_prase = 0
        for layer_index, dilation in enumerate(self.dilations):
            var = self.variables['dilated_stack'][layer_index]
            self.prune(sess,var['filter'],threshold,'dilated_stack' + str(layer_index) + 'filter')
            self.prune(sess, var['gate'], threshold, 'dilated_stack' + str(layer_index) + 'gate')
            self.prune(sess, var['lcond_filter'], threshold, 'dilated_stack' + str(layer_index) + 'lcond_filter')
            self.prune(sess, var['lcond_gate'], threshold, 'dilated_stack' + str(layer_index) + 'lcond_gate')
            self.prune(sess, var['dense'], threshold, 'dilated_stack' + str(layer_index) + 'dense')
            self.prune(sess, var['skip'], threshold, 'dilated_stack' + str(layer_index) + 'skip')
        var = self.variables['postprocessing']
        self.prune(sess, var['postprocess1'], threshold, 'post1')
        self.prune(sess, var['postprocess2'], threshold, 'post2')
        return self.c_prase
    def _trainablevar2sparsetensor(self,sess):
        #trainable = tf.trainable_variables()
        #self.cread_placehold()
        #compression_num = self.thread_prune(sess, threshold=1)
        self.c_prase=0
        for layer_index, dilation in enumerate(self.dilations):
            var = self.variables['dilated_stack'][layer_index]
            self.sparse_params['dilated_stack'+str(layer_index)+'filter0']=self._var2sparse(sess,tf.transpose(var['filter'][0,:,:]))
            self.sparse_params['dilated_stack' + str(layer_index) + 'gate0'] = self._var2sparse(sess, tf.transpose(var['gate'][0,:,:]))
            self.sparse_params['dilated_stack' + str(layer_index) + 'lcond_filter'] = self._var2sparse(sess, tf.transpose(var['lcond_filter'][0,:,:]))
            self.sparse_params['dilated_stack' + str(layer_index) + 'lcond_gate'] = self._var2sparse(sess, tf.transpose(var['lcond_gate'][0,:,:]))
            self.sparse_params['dilated_stack'+str(layer_index)+'filter1']=self._var2sparse(sess,tf.transpose(var['filter'][1,:,:]))
            self.sparse_params['dilated_stack' + str(layer_index) + 'gate1'] = self._var2sparse(sess, tf.transpose(var['gate'][1,:,:]))
            self.sparse_params['dilated_stack' + str(layer_index) + 'dense'] = self._var2sparse(sess, tf.transpose(var['dense'][0,:,:]))
            self.sparse_params['dilated_stack' + str(layer_index) + 'skip'] = self._var2sparse(sess, tf.transpose(var['skip'][0,:,:]))
        var = self.variables['postprocessing']
        self.sparse_params['post1'] = self._var2sparse(sess,tf.transpose(var['postprocess1'][0,:,:]))
        self.sparse_params['post2'] = self._var2sparse(sess, tf.transpose(var['postprocess2'][0,:,:]))

        print (self.c_prase)
        print(self.c_params)

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation, 
                               global_condition = None, local_condition = None):
        '''Creates a single causal dilated convolution layer.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output.
        '''
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if global_condition is not None:
            weights_gcond_filter = variables['gcond_filter']
            weights_gcond_gate = variables['gcond_gate']
            conv_filter = conv_filter + \
                tf.matmul(global_condition, weights_gcond_filter)
            conv_gate = conv_gate + \
                tf.matmul(global_condition, weights_gcond_gate)
            
        if local_condition is not None:
            weights_lcond_filter = variables['lcond_filter']
            weights_lcond_gate = variables['lcond_gate']
            
            conv_filter = conv_filter + \
            tf.nn.conv1d(local_condition, weights_lcond_filter, stride=1, padding="SAME")
            conv_gate = conv_gate + \
                        tf.nn.conv1d(local_condition, weights_lcond_gate, stride=1, padding="SAME")
                
        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        layer = 'layer{}'.format(layer_index)
        tf.histogram_summary(layer + '_filter', weights_filter)
        tf.histogram_summary(layer + '_gate', weights_gate)
        tf.histogram_summary(layer + '_dense', weights_dense)
        tf.histogram_summary(layer + '_skip', weights_skip)
        if global_condition is not None:
            tf.histogram_summary(layer + '_gc_filter', weights_gcond_filter)
            tf.histogram_summary(layer + '_gc_gate', weights_gcond_gate)
        if local_condition is not None:
            tf.histogram_summary(layer + '_lc_filter', weights_lcond_filter)
            tf.histogram_summary(layer + '_lc_gate', weights_lcond_gate)
        if self.use_biases:
            tf.histogram_summary(layer + '_biases_filter', filter_bias)
            tf.histogram_summary(layer + '_biases_gate', gate_bias)
            tf.histogram_summary(layer + '_biases_dense', dense_bias)
            tf.histogram_summary(layer + '_biases_skip', skip_bias)

        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, past_weights, curr_weights):

        # TODO generalize to filter_width > 2
        # past_weights = weights[0, :, :]
        # curr_weights = weights[1, :, :]
        output = tf.sparse_tensor_dense_matmul(past_weights, state_batch) + tf.sparse_tensor_dense_matmul(
            curr_weights, input_batch)
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation, global_condition, local_condition):

        variables = self.variables['dilated_stack'][layer_index]








        output_filter = self._generator_conv(
            input_batch, state_batch, self.sparse_params['dilated_stack' + str(layer_index) + 'filter0'],
            self.sparse_params['dilated_stack' + str(layer_index) + 'filter1'])
        output_gate = self._generator_conv(
            input_batch, state_batch, self.sparse_params['dilated_stack' + str(layer_index) + 'gate0'],
            self.sparse_params['dilated_stack' + str(layer_index) + 'gate1'])

        if local_condition is not None:
            weights_lcond_filter = self.sparse_params['dilated_stack' + str(layer_index) + 'lcond_filter']
            weights_lcond_gate = self.sparse_params['dilated_stack' + str(layer_index) + 'lcond_gate']

            output_filter = output_filter + \
                            tf.sparse_tensor_dense_matmul(weights_lcond_filter, local_condition)
            output_gate = output_gate + \
                          tf.sparse_tensor_dense_matmul(weights_lcond_gate, local_condition)

        if self.use_biases:
            output_filter = output_filter + tf.reshape(variables['filter_bias'], [-1, 1])
            output_gate = output_gate + tf.reshape(variables['gate_bias'], [-1, 1])

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = self.sparse_params['dilated_stack' + str(layer_index) + 'dense']
        transformed = tf.sparse_tensor_dense_matmul(weights_dense, out)
        if self.use_biases:
            transformed = transformed + tf.reshape(variables['dense_bias'], [-1, 1])

        weights_skip = self.sparse_params['dilated_stack' + str(layer_index) + 'skip']
        skip_contribution = tf.sparse_tensor_dense_matmul(weights_skip, out)
        if self.use_biases:
            skip_contribution = skip_contribution + tf.reshape(variables['skip_bias'], [-1, 1])

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch, global_condition = None, 
                        local_condition = None):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch
        global_condition = None

        fil=self.variables['upsample']['filter']

        up_lc=tf.nn.conv2d_transpose(tf.reshape(local_condition,[1,1,-1,self.local_channels]),fil,[1,1,self.upsample_rate*tf.shape(local_condition)[1],self.output_channel],[1,1,self.upsample_rate,1],padding='VALID')
        #current_layer = self._create_causal_layer(current_layer)
        up_lc=tf.reshape(up_lc, [1, -1, self.output_channel])
        up_w = self.variables['upsample']['filter1']
        up_lc1 = tf.nn.conv1d(up_lc,up_w,stride=1,padding="SAME")
        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation, global_condition,up_lc1+up_lc
                        )
                    if layer_index==0:
                        total = output
                    else:
                        total =total+output


        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            tf.histogram_summary('postprocess1_weights', w1)
            tf.histogram_summary('postprocess2_weights', w2)
            if self.use_biases:
                tf.histogram_summary('postprocess1_biases', b1)
                tf.histogram_summary('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            #total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)

        return conv2

    def _create_generator(self, input_batch, global_condition=None, local_condition=None):

        init_ops = []
        push_ops = []
        outputs = []
        current_layer = tf.transpose(input_batch)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    q = tf.FIFOQueue(
                        dilation,
                        dtypes=tf.float32,
                        shapes=(self.batch_size, self.residual_channels), name='queue2')
                    init = q.enqueue_many(
                        tf.zeros((dilation, self.batch_size,
                                  self.residual_channels)))

                    current_state = q.dequeue()
                    push = q.enqueue([tf.transpose(current_layer)])
                    init_ops.append(init)
                    push_ops.append(push)

                    output, current_layer = self._generator_dilation_layer(
                        current_layer, tf.transpose(current_state), layer_index, dilation,
                        global_condition, tf.transpose(local_condition))
                    outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            variables = self.variables['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.sparse_params['post1']
            w2 = self.sparse_params['post2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)

            conv1 = tf.sparse_tensor_dense_matmul(w1, transformed1)
            if self.use_biases:
                conv1 = conv1 + tf.reshape(b1, [-1, 1])
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.sparse_tensor_dense_matmul(w2, transformed2)
            if self.use_biases:
                conv2 = conv2 + tf.reshape(b2, [-1, 1])

        return tf.transpose(conv2)
    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _embed_input(self, input_batch):
        '''Looks up the embeddings of the the waveform amplitudes.
        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('input_embedding'):
            embedding_table = self.variables['embeddings']['input_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table,
                                               input_batch)
            shape = [self.batch_size, -1, self.residual_channels]
            embedding = tf.reshape(embedding, shape)
        return embedding


    def predict_proba(self, waveform, global_condition = None, 
                      local_condition = None, tts_feature=None,num_samples=None,name='wavenet'):
        '''Computes the probability distribution of the next sample based on
        all samples in the input waveform.
        If you want to generate audio by feeding the output of the network back
        as an input, see predict_proba_incremental for a faster alternative.'''
        with tf.name_scope(name):
            if self.scalar_input:
                encoded = tf.cast(waveform, tf.float32)
                encoded = tf.reshape(encoded, [-1, 1])
            else:
                encoded = self._one_hot(waveform)

            raw_output = self._create_network(encoded, global_condition, 
                local_condition)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])
            # Cast to float64 to avoid bug in TensorFlow
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            last = tf.reshape(last, [-1])
            return last

    def predict_proba_incremental(self, waveform, global_condition = None, 
                                  local_condition = None, name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not "
                                      "support scalar input yet.")
        with tf.name_scope(name):

            #encoded = tf.one_hot(waveform, self.quantization_channels)
            #encoded = tf.reshape(encoded, [-1, self.quantization_channels])
            encoded = self._embed_input(waveform)
            encoded = tf.reshape(encoded,[-1,self.residual_channels])
            raw_output = self._create_generator(encoded, global_condition, local_condition)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            out = tf.reshape(last, [-1])



            return out



    def genrator_upsampling_tts(self,local_condition = None):
        fil = self.variables['upsample']['filter']

        up_lc = tf.nn.conv2d_transpose(tf.reshape(local_condition, [1, 1, -1, self.local_channels]), fil,
                                       [1, 1, self.upsample_rate * tf.shape(local_condition)[0], self.output_channel],
                                       [1, 1, self.upsample_rate, 1], padding='VALID')
        up_lc=tf.reshape(up_lc, [1, -1, self.output_channel])
        up_w = self.variables['upsample']['filter1']
        up_lc1 = tf.nn.conv1d(up_lc,up_w,stride=1,padding="SAME")
        
        return tf.reshape(up_lc+up_lc1,[-1,self.output_channel])
    def loss(self,
             input_batch,
             l2_regularization_strength=None,
             global_condition=None,
             local_condition=None,
             name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.name_scope(name):
            # We mu-law encode and quantize the input audioform.
            input_encode = mu_law_encode(input_batch,
                                        self.quantization_channels)

            #encoded = self._one_hot(input_encode)
            if self.scalar_input:
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = input_encode

            if global_condition is not None:
                gc_encoded = tf.one_hot(global_condition, self.global_channels)
            else:
                gc_encoded = None
                
            if local_condition is not None:

                lc_encoded = local_condition
            else:
                lc_encoded = None
            network_input1 = self._embed_input(network_input)
            raw_output = self._create_network(network_input1,
                global_condition=gc_encoded,
                local_condition=lc_encoded)

            with tf.name_scope('loss'):
                # Shift original input left by one sample, which means that
                # each output sample has to predict the next input sample.
                encoded = self._one_hot(input_encode)
                shifted = tf.slice(encoded, [0, 1, 0],
                                   [-1, tf.shape(encoded)[1] - 1, -1])
                shifted = tf.pad(shifted, [[0, 0], [0, 1], [0, 0]])

                prediction = tf.reshape(raw_output,
                                        [-1, self.quantization_channels])
                loss = tf.nn.softmax_cross_entropy_with_logits(
                    prediction,
                    tf.reshape(shifted, [-1, self.quantization_channels]))
                reduced_loss = tf.reduce_mean(loss)

                tf.scalar_summary('loss', reduced_loss)

                if l2_regularization_strength is None:
                    return reduced_loss
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                        for v in tf.trainable_variables()
                                        if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss +
                                  l2_regularization_strength * l2_loss)

                    tf.scalar_summary('l2_loss', l2_loss)
                    tf.scalar_summary('total_loss', total_loss)

                    return total_loss
