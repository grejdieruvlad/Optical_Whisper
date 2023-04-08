import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 
from scipy.io.wavfile import read,write
import sounddevice as sd
import wavio as wv
import glob

class Audio:
    def record_audio_stream(self):
        print() 
    def record_speaker_voice_sample_for_traning(self, new_speaker_name, stream):
          Name = new_speaker_name
          FORMAT= pyaudio.paInt16
          CHANNELS= 1
          RATE=16000
          audio = pyaudio.PyAudio()
          Recordframes = stream
          all_training_files = glob.glob("./voice_recognition/training_set/" + Name + "*")
          index = []
          last_index = 0
          if len(all_training_files) > 0:
               for each in all_training_files:
                   index.append(each.split("/")[-1].split("-")[1].split("sample")[1].split(".")[0])
               last_index = int(max(index)) + 1
          OUTPUT_FILENAME=Name+"-sample" + str(last_index) +".wav"
          WAVE_OUTPUT_FILENAME=os.path.join("./voice_recognition/training_set/",OUTPUT_FILENAME)
          trainedfilelist = open("./voice_recognition/training_set_addition.txt", 'a')
          trainedfilelist.write(OUTPUT_FILENAME+"\n")
          waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
          waveFile.setnchannels (CHANNELS)
          waveFile.setsampwidth(2)
          waveFile.setframerate (RATE)
          waveFile.writeframes (b''.join(Recordframes))
          waveFile.close()
    def record_speaker_voice_sample_for_verifying(self, actual_speaker, stream):
          Name = actual_speaker.name
          FORMAT= pyaudio.paInt16
          CHANNELS= 1
          RATE=16000
          audio = pyaudio.PyAudio()
          Recordframes = stream
          OUTPUT_FILENAME=Name+"-sample"".wav"
          WAVE_OUTPUT_FILENAME=os.path.join("./voice_recognition/testing_set/",OUTPUT_FILENAME)
          testingfilelist = open("./voice_recognition/testing_set_addition.txt", 'w')
          testingfilelist.write(OUTPUT_FILENAME+"\n")
          waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
          waveFile.setnchannels (CHANNELS)
          waveFile.setsampwidth(2)
          waveFile.setframerate (RATE)
          waveFile.writeframes (b''.join(Recordframes))
          waveFile.close()
    def calculate_delta(self, array):
        rows, cols= array.shape
     #    print(rows)
     #    print(cols)
        deltas = np.zeros((rows, 20))
        N = 2
        for i in range (rows):
             index = []
             j = 1
             while j <= N:
                  if i-j < 0:
                       first =0
                  else:
                       first = i-j
                  if i+j > rows-1:
                       second = rows-1
                  else:
                       second = i+j
                  index.append((second, first))
                  j+=1
             deltas[i] = (array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
        return deltas
    def extract_features (self,audio,rate):
        mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20, nfft = 1200, appendEnergy = True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        # print(mfcc_feature)
        delta = self.calculate_delta (mfcc_feature)
        combined = np.hstack((mfcc_feature, delta))
        return combined

    def train_voice_recognition_model(self):
        source = "./voice_recognition/training_set/" #training set path
        dest = "./voice_recognition/trained_models/" #destination path to store trained models
        train_file = "./voice_recognition/training_set_addition.txt" #training file wiith sample names
        file_paths = open(train_file, 'r')
        count = 1
        features = np.asarray(())
        for path in file_paths:
             path = path.strip()
             sr, audio = read(source + path)
             vector = self.extract_features(audio, sr)
             if features.size == 0:
                  features = vector
             else:
                  features = np.vstack((features, vector))
             if count == 1:
                  gmm = GaussianMixture(n_components = 6, max_iter = 200, covariance_type='diag',n_init = 3)
                  gmm.fit(features)
                  # dumping the trained gaussian model
                  picklefile = path.split("-")[0]+".gmm"
                  pickle.dump (gmm, open(dest + picklefile, 'wb'))
                  print('+ modeling completed for speaker: ',picklefile," with data point = ",features.shape)
                  features
                  np.asarray(())
                  count = 0
             count = count + 1
    def verify_speaker_by_voice(self):
        source = "./voice_recognition/testing_set/" # Path of test samples
        modelpath = "./voice_recognition/trained_models/" # path for trained models
        test_file = "./voice_recognition/testing_set_addition.txt" #test samples names
        file_paths = open(test_file, 'r')
        gmm_files = [os.path.join(modelpath, fname) for fname in
                     os.listdir (modelpath) if fname.endswith('.gmm')]
        #Load the Gaussian gender Models
        models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
        speakers = [fname.split("/")[-1].split(".gmm")[0] for fname
                   in gmm_files]
        # Read the test directory and get the list of test audio files
        for path in file_paths:
             path = path.strip()
             sr, audio = read(source + path)
             vector = self.extract_features (audio, sr)
             log_likelihood = np.zeros(len(models))
             for i in range(len(models)):
                  gmm = models[i] #checking with each model one by one
                  scores = np.array(gmm.score(vector))
                  log_likelihood[i] = scores.sum()
             winner = np.argmax(log_likelihood)
             print("\tdetected as - ", speakers [winner])
             return speakers [winner]
           
           
# for count in range(5):
          # print(" ---------------------------------- record device list ---------------")
          # info = audio.get_host_api_info_by_index(0)
          # numdevices = info.get('deviceCount')
          # for i in range(0, numdevices):
          #      if (audio.get_device_info_by_host_api_device_index(0, 1).get('maxInputChannels')) > 0:
          #           print("Input Device id ", i, "-", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
          # print("----------------------------------------------------------------------") 
          # index = int(input())
          # print("recording via index "+str(device_index))
          # stream = audio.open(format=FORMAT, channels=CHANNELS,
          #          rate=RATE, input=True, input_device_index = device_index,
          #          frames_per_buffer=CHUNK)
          # print ("recording started")
 #Name of the speaker we can take from actual_speaker value
          #Name = (input("Please Enter Your Name:"))

          # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
          #      data = stream.read(CHUNK)
          #      Recordframes.append(data)
          # print ("recording stopped")
          # stream.stop_stream()
          # stream.close()    
          # audio.terminate()
# a = Audio()
# a.verify_speaker_by_voice()
