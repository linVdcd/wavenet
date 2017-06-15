from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf
import time
from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader

SAMPLES = 16000
LOGDIR = './logdir'
WINDOW = 8000
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0.1


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', default= './logdir/train/2016-11-04T11-55-50/model.ckpt-88000',type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--logdir',
        type=str,
        default='./logdir/train/2016-11-02T18-08-18',
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--window',
        type=int,
        default=WINDOW,
        help='The number of past samples to take into '
        'account at each step')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default='out.wav',
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default='../wavenet_data/wav/hs_zh_arctic_hmb_10325.wav',
        help='The wav file to start generation from')
    parser.add_argument(
        '--speaker_id',
        type=int,
        default=None,
        help='The id of the speaker (model must have trained with global \
            conditioning)'
    )
    parser.add_argument(
        '--speaker_text',
        type=str,
        default='../wavenet_data_v2/lab_frame_digit/hs_zh_arctic_bjs_00553.lab',#'../wavenet_data/lab_frame_digit/hs_zh_arctic_bjs_00553.lab',
        help='Sample text to be spoken'
    )
    return parser.parse_args()


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size=WINDOW,
                silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
            lambda: tf.size(quantized),
            lambda: tf.constant(window_size))

    return quantized[:cut_index]


def main():
    args = get_arguments()
    p, labname = os.path.split(args.speaker_text)
    p, modelname = os.path.split(args.checkpoint)
    args.wav_out_path = p+'/' + labname[:-4] + '_model' + modelname[11:] + '.wav'
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'],
        local_channels=wavenet_params['local_channels'],
        global_channels = wavenet_params['global_channels'],
	    output_channels=wavenet_params["upsample_channels"]
	)

    quantization_channels = wavenet_params['quantization_channels']
    samples = tf.placeholder(tf.int32)

    seedlen = 0
    if args.wav_seed:
        seed = create_seed(args.wav_seed,
                           wavenet_params['sample_rate'],
                           quantization_channels)

        waveform = sess.run(seed).tolist()
        seedlen =8000
    else:
        waveform = [0]

    if args.speaker_id:
        id_embedded = tf.one_hot([args.speaker_id], net.global_channels, 
            axis=-1)
    else:
        id_embedded = None

    if args.speaker_text:
        if args.wav_seed:
            _, seedname = os.path.split(args.wav_seed)

        tts_feature = np.loadtxt(args.speaker_text)

        if args.wav_seed:
            tts_seedpath = os.path.join('../wavenet_data_v2/', 'lab_frame_digit/' + seedname[:-4] + '.lab')  # wav seed
            tts_seedfeature = np.loadtxt(tts_seedpath)# wav seed

            tts_feature = np.append(tts_seedfeature[0:seedlen/40][:], tts_feature,axis=0)  # wav seed

        min_max = np.loadtxt('../wavenet_data_v2/min-max.txt')
        tts_feature = (tts_feature - min_max[0, :]) / (min_max[1, :] - min_max[0, :] + 0.001)
        labpath, name = os.path.split(args.speaker_text)

        f0path = os.path.join('../wavenet_data_v2/', 'lf0_digit/' + name[:-4] + '.lf0')
        f0 = np.loadtxt(f0path)
        if args.wav_seed:
            f0seedpath = os.path.join('../wavenet_data_v2/', 'lf0_digit/' + seedname[:-4] + '.lf0')
            f0seed = np.loadtxt(f0seedpath)
            f0 = np.hstack((f0seed[0:seedlen/40], f0))
        min_max = np.exp(np.loadtxt('../wavenet_data_v2/lf0-min-max.txt'))
        f0 = (np.exp(f0) - min_max[0]) / (min_max[1] - min_max[0])
        lenlab = tts_feature.shape[0]
        lenf0 = f0.shape[0]
        if lenlab > lenf0:
            tts_feature = tts_feature[:lenf0, :]
        else:
            f0 = f0[:lenlab]

        tts_feature = np.column_stack((tts_feature, f0))
        # lab = lab.astype('float')
        #tts_feature = np.repeat(tts_feature, int(240 * wavenet_params['sample_rate'] / 48000), axis=0)
        #args.samples = tts_feature.shape[0]
        # tts_feature = np.reshape(tts_feature,(1,args.samples,556))
        lc = tf.placeholder(dtype=tf.float32,shape=None)
        upsample_tts_feature=net.genrator_upsampling_tts(local_condition=lc)


        text_embedded = tf.placeholder(dtype=tf.float32, shape=(1, wavenet_params['upsample_channels']))
        text_embedded = tf.reshape(text_embedded, [-1, wavenet_params['upsample_channels']])
        # text_embedded[0,:] = tts_feature[0,:]
    else:
        text_embedded = None

    variables_to_restore = {
        var.name[:-2]: var for var in tf.all_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    #print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)
    net._trainablevar2sparsetensor(sess)
    if args.fast_generation:
        next_sample = net.predict_proba_incremental(samples, id_embedded,
            text_embedded)
    else:
        next_sample = net.predict_proba(samples, id_embedded, text_embedded)


    if args.fast_generation:
        sess.run(tf.initialize_all_variables())
        sess.run(net.init_ops)




    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    decode = mu_law_decode(samples, wavenet_params['quantization_channels'])
    print('upsampling tts feature!')
    local_condition = sess.run(upsample_tts_feature, feed_dict={lc: tts_feature}).tolist()
    print('done')


    local_condition = np.array(local_condition)
    args.samples =local_condition.shape[0]






    last_sample_timestamp = datetime.now()
    s = time.time()
    for step in range(seedlen,args.samples):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window = waveform[-1]
        else:
            if len(waveform) > args.window:
                window = waveform[-args.window:]
            else:
                window = waveform
            outputs = [next_sample]

        # Run the WaveNet to predict the next sample.

        prediction = sess.run(outputs, feed_dict={samples: window,text_embedded:local_condition[step:step+1,:]})[0]
        sample = np.random.choice(
            np.arange(quantization_channels), p=prediction)
        #print(sample)
        waveform.append(sample)

        # Show progress only once per second.
        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

        # If we have partial writing, save the result so far.
        if (args.wav_out_path and args.save_every and
                (step + 1) % args.save_every == 0):
            out = sess.run(decode, feed_dict={samples: waveform})
            write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    # Introduce a newline to clear the carriage return from the progress.
    e = time.time()
    print(e-s)

    # Save the result as an audio summary.
    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.train.SummaryWriter(logdir)
    tf.audio_summary('generated', decode, wavenet_params['sample_rate'])
    summaries = tf.merge_all_summaries()
    summary_out = sess.run(summaries,
                           feed_dict={samples: np.reshape(waveform, [-1, 1])})
    writer.add_summary(summary_out)

    # Save the result as a wav file.

    if args.wav_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})
        write_wav(out[seedlen:], wavenet_params['sample_rate'], args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
