# from array import array
# from struct import pack
# import sys, os
# import time
# import pyaudio
# import wave
# import gobject
# import subprocess
# import contextlib
# import threading

# import pygtk
# pygtk.require("2.0")
# import gtk
# class A:
#     def run(self):
#         print ("Run")
#         self.running = True
#         decoder = Decoder(self.get_config())        #Create the decoder from the config
#         p = pyaudio.PyAudio()
#         stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
#         stream.start_stream()
#         in_speech_bf = True         #Needed to get the state, when you are speaking/not speaking -> statusbar
#         decoder.start_utt()
#         while self.running:
#             buf = stream.read(1024)          #Read the first Chunk from the microphone
#             if buf:
#                 #Pass the Chunk to the decoder
#                 decoder.process_raw(buf, False, False)
#                 try:
#                     #If the decoder has partial results, display them in the GUI.
#                     if  decoder.hyp().hypstr != '':
#                         hypstr = decoder.hyp().hypstr
#                         print('Partial decoding result: '+ hypstr)
#                         if textbuffer_partial.get_text(*textbuffer_partial.get_bounds()) != hypstr:
#                             gtk.gdk.threads_enter()
#                             textbuffer_partial.set_text(hypstr)
#                             gtk.gdk.threads_leave()
#                 except AttributeError:
#                     pass
#                 if decoder.get_in_speech():
#                     pass
#                     #sys.stdout.write('.')
#                     #sys.stdout.flush()
#                 if decoder.get_in_speech() != in_speech_bf:
#                     in_speech_bf = decoder.get_in_speech()
#                     #When the speech ends:
#                     if not in_speech_bf:
#                         decoder.end_utt()
#                         try:
#                             #Since the speech is ended, we can assume that we have final results, then display them
#                             if decoder.hyp().hypstr != '':
#                                 decoded_string = decoder.hyp().hypstr
#                                 print('Stream decoding result: '+ decoded_string)
#                                 gtk.gdk.threads_enter()
#                                 textbuffer_end.insert( textbuffer_end.get_end_iter(), decoded_string+"\n")
#                                 gtk.gdk.threads_leave()
#                         except AttributeError:
#                             pass
#                         decoder.start_utt()            #Say to the decoder, that a new "sentence" begins
#                         gtk.gdk.threads_enter()
#                         statusbar.push(0, "Listening: No audio")
#                         gtk.gdk.threads_leave()
#                         print("stopped listenning")
#                     else:
#                         gtk.gdk.threads_enter()
#                         statusbar.push(0, "Listening: Incoming audio...")
#                         gtk.gdk.threads_leave()
#                         print("start listening")
#             else:
#                 break
#             #print decoder.get_in_speech()
#         #We get here, out of the while loop, if the user aborts the recognition thread.
#         decoder.end_utt()
#         print("PS  is over.")
#         #Do some cleanup
#         stream.stop_stream()
#         stream.close()
#         p.terminate()
# a = A()
# a.run()



import glob
# Name = "Doina"
# a = glob.glob("./voice_recognition/training_set/" + Name + "*")
# print("./voice_recognition/training_set/" + Name + "*")
# for each in a:
#     index = each.split("/")[-1].split("-")[1].split("sample")[1].split(".")[0]
# last_index = max(index)
# print(last_index)
    #   os.path.join("./voice_recognition/trainindg_set/

all_training_files = glob.glob("./voice_recognition/training_set/" + "Doina" + "*")
print(all_training_files)
index = []
for each in all_training_files:
    index.append(each.split("/")[-1].split("-")[1].split("sample")[1].split(".")[0])
print(index)
last_index = max(index)