import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    if files==[]:
        print 'gen:havenot wav files'
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename


def load_vctk_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the VCTK dataset, and
    additionally the ID of the corresponding speaker.'''
    files = find_files(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        matches = speaker_re.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]
        
        dirs_vctk_wav48_p, name = os.path.split(filename)
        dirs_vctk_wav48, p = os.path.split(dirs_vctk_wav48_p)
        dirs_vctk, wav48 = os.path.split(dirs_vctk_wav48)
        filename_text = os.path.join(dirs_vctk, 'txt', p, name[:-4] + '.txt')
        
        with open(filename_text) as f:
            text = f.read()
        yield audio, (filename, speaker_id, list(text))

def load_tts_audio(directory,samle_rate):
    files = find_files(directory)
    for filename in files:
        #print filename
        audio,_ = librosa.load(filename,sr = samle_rate,mono=True)
        audio = audio.reshape(-1,1)
        dirs_wav,name = os.path.split(filename)
        dirs,_ = os.path.split(dirs_wav)

        filename_lab = os.path.join(dirs,'lab_frame_digit/',name[:-4]+'.lab')
        filename_f0 = os.path.join(dirs,'lf0_digit/',name[:-4]+'.lf0')
        lab = np.loadtxt(filename_lab)
        min_max = np.loadtxt('../wavenet_data_v2/min-max.txt')
        lab = (lab-min_max[0,:])/(min_max[1,:]-min_max[0,:]+0.001)

        f0 =  np.loadtxt(filename_f0)

        min_max = np.exp(np.loadtxt('../wavenet_data_v2/lf0-min-max.txt'))
        f0 = (np.exp(f0) - min_max[0])/(min_max[1]-min_max[0])
        lenlab = lab.shape[0]
        lenf0 = f0.shape[0]
        if lenlab>lenf0:
            lab = lab[:lenf0,:]
        else:
            f0 = f0[:lenlab]
        #f0 = np.repeat(f0, lab.shape[0])
        #print lab.shape[0]
        #print f0.shape[0]
        #print audio.shape[0]
        lab = np.column_stack((lab, f0))

        #lab = np.repeat(lab, int(240*samle_rate/48000), axis=0)
        lenwav = audio.shape[0]
        lenlab = lab.shape[0]*(240*samle_rate/48000)
        if lenwav>lenlab:
            audio = audio[:lenlab,0]
        else:
            lab  =lab[:-1,:]
            audio = audio[:lenlab-240*samle_rate/48000,0]

        lab = lab.astype('float')
        yield audio,lab
def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=256,
                 vctk=False,
                 tts = False,
                 ttsfeature_size=556):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.vctk = vctk
        self.tts = tts
        self.ttsfeature_size=ttsfeature_size
        self.threads = []
        
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size, ['float32'],
                                         shapes=[(None, 1)])
                                         
        self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
        self.text_placeholder = tf.placeholder(dtype=tf.string, shape=(None,))

        self.conditions_queue = tf.PaddingFIFOQueue(queue_size, 
                                         ['int32', 'string'],
                                         shapes=[(), (None,)])
        self.tts_placeholder = tf.placeholder(dtype = tf.float32,shape=None)
        self.tts_conditions_queue = tf.PaddingFIFOQueue(queue_size,['float32'],shapes=[(None, ttsfeature_size)])

        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        self.conditions_enqueue = self.conditions_queue.enqueue([
                                                    self.id_placeholder, 
                                                    self.text_placeholder])

        self.tts_conditions_enqueue = self.tts_conditions_queue.enqueue([self.tts_placeholder])


    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output
        
    def conditions_dequeue(self, num_elements):
        output = self.conditions_queue.dequeue_many(num_elements)
        return output
    def tts_conditions_dequeue(self,num_elements):
        output = self.tts_conditions_queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        buffer_lab = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            if self.vctk:
                iterator = load_vctk_audio(self.audio_dir, self.sample_rate)
            elif self.tts:
                iterator = load_tts_audio(self.audio_dir,self.sample_rate)
            else:
                iterator = load_generic_audio(self.audio_dir, self.sample_rate)
            for audio, extra in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(extra))

                

                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, audio)
                    if self.tts:
                        lab = extra
                        buffer_lab = np.append(buffer_lab,lab)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        buffer_ = buffer_[self.sample_size:]
                        if self.vctk:
                            filename, user_id, text = extra
                            sess.run(self.conditions_enqueue,
                                     feed_dict = {self.id_placeholder: user_id,
                                                  self.text_placeholder: text})
                        if self.tts:
                            piece_lab = np.reshape(buffer_lab[:self.sample_size*self.ttsfeature_size/40], [self.sample_size/40, self.ttsfeature_size])
                            sess.run(self.tts_conditions_enqueue,feed_dict = {self.tts_placeholder:piece_lab})
                            buffer_lab = buffer_lab[self.sample_size*self.ttsfeature_size/40:]
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: np.reshape(audio, (-1,1))})
                    if self.vctk:
                        filename, user_id, text = extra
                        sess.run(self.conditions_enqueue,
                                 feed_dict = {self.id_placeholder: user_id,
                                              self.text_placeholder: text})
                    if self.tts:
                        lab = extra
                        sess.run(self.tts_conditions_enqueue, feed_dict={self.tts_placeholder: lab})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
