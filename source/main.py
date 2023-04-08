import chunk
from email.mime import audio
from glob import glob
from html.entities import name2codepoint
from multiprocessing.pool import TERMINATE
from multiprocessing.spawn import import_main_path
from tkinter.messagebox import NO
from traceback import print_tb
import speech_recognition as sr 
import cv2 
import os 
import sys
import time
import re
from playsound import playsound as ps
from PIL import Image
import numpy as np
from Audio import Audio
sys.path.append(".")
import fire_alarm_detection as alarm_detection


from Speaker import Speaker 

from threading  import Thread, excepthook
from re import search
key = 'o'
message = ""
name_subtring = [
    "my name is ", 
    # "my name's", 
    "I am " ,
    "I'm ",
    "can call me "
    ]
bye_substring = [
    "Bye",
    "Goodbye",
    "Bye-by",
    "Farewell",
    "Cheerio",
    "See you",
    "I’m out",
    "Take care",
    "Take it easy",
    "I’m off",
    "Gotta go!",
    "Good night",
    "Bye for now",
    "See you later",
    "Keep in touch",
    "Catch you later",
    "See you soon",
    "I gotta take off",
    "Talk to you later",
    "See you next time",
    "Have a good one",
    "Have a good day",
    "I’ve got to get going",
    "I must be going"
    ]

stream_segment = []
# last_speaker 

error_nr = 2

def register_new_face(name, gray1, faces1, img1, cam1, id):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height
        face_detector = cv2.CascadeClassifier('./face_recognition/haarcascade_frontalface_default.xml')
        # For each person, enter one numeric face id
        #face_id = input('\n enter user id end press <return> ==>  ')
        face_id = id
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
        count = 0
        while(True):
            ret, img = cam.read()
            img = cv2.flip(img, 1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1
                # Save the captured image into the datasets folder
                cv2.imwrite("./face_recognition/dataset/User." + str(face_id) + '.' +  
                            str(count) + ".jpg", gray[y:y+h,x:x+w])
            print("\n [INFO] Face registred successfull")
            if count == 30:
                break
            info = "Recording face sample for speaker: "
            cv2.putText(img, info + name , (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('camera', img)
            k = cv2.waitKey(1) # Press 'ESC' for exiting video
            if key == 'q':
                break
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                break
        # Do a bit of cleanup
        #cam.release()
        # cv2.destroyAllWindows()
        # cv2.destroyWindow('image')
def train_algorithm():    
    # Path for face image database
    path = './face_recognition/dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("./face_recognition/haarcascade_frontalface_default.xml")
    # function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('./face_recognition/trainer/trainer.yml') 
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

class Video():
    img_counter = 0
    frame = None
    video = None
    global stream_segment
    global key
    def recording_video(self):
        global emergency
        global message
        global last_speaker
        global names
        global key
        global actual_speaker
        global speakers
        global message_start_time
        global info
        global _audio
        global face_status
        global tmp_speaker
        cascadePath = "./face_recognition/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cam = cv2.VideoCapture(0)
        cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("camera", 1280 , 800)
        cam.set(3, 800) # set video widht
        cam.set(4, 600) # set video height
        
        # Define min window size to be recognized as a face
        minW = 0.2*cam.get(3)
        minH = 0.2*cam.get(4)
        if not os.path.exists('./face_recognition/trainer/trainer.yml'):
            while True:
                ret, img = cam.read()
                img = cv2.flip(img, 1) # Flip vertically
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                
                # faces = faceCascade.detectMultiScale( 
                #     gray,
                #     scaleFactor = 1.2,
                #     minNeighbors = 5,
                #     minSize = (int(minW), int(minH)),
                #    )
                # for(x,y,w,h) in faces:
                #     id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                #     cv2.putText(img, str(id) , (x+7,y-5), font, 1, (0,255,0), 1)
                cv2.imshow('camera',img)
                if len(actual_speaker.message) > 40 and actual_speaker.message_status:
                    message1 = actual_speaker.message[0:40]
                    message2 = actual_speaker.message[40:]
                    if len(last_speaker.message) > 0 or (last_speaker.name != "Unknown") :
                        cv2.putText(img,  last_speaker.name + ":" +last_speaker.message , (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_speaker.text_color, 2, cv2.LINE_AA)
                    cv2.putText(img, actual_speaker.name +":" + message1, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, actual_speaker.text_color, 2, cv2.LINE_AA)
                    cv2.putText(img, message2, (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, actual_speaker.text_color, 2, cv2.LINE_AA)
                elif len(actual_speaker.message) < 40 and len(actual_speaker.message) > 0  and actual_speaker.message_status:
                    if len(last_speaker.message) > 0 or (last_speaker.name != "Unknown") :
                        cv2.putText(img,  last_speaker.name + ":" +last_speaker.message , (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_speaker.text_color, 2, cv2.LINE_AA)
                    cv2.putText(img, actual_speaker.name +":" + actual_speaker.message, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, actual_speaker.text_color, 2, cv2.LINE_AA)
                    # aici vom vedea care este situatia.
                    # last_speaker.message = actual_speaker.message
                if time.time() - message_start_time > 5 :
                    actual_speaker.message_status = False
                cv2.imshow('camera',img) 
                for each in name_subtring:
                        if each in  actual_speaker.message.lower().strip():
                            index = actual_speaker.message.find(each)
                            tmp = actual_speaker.message[index+len(each):]
                            tmp_name = re.split(';. |, |\*|\n',tmp)[0]
                            name = tmp_name.split('.')[0].strip()
                            info = "Recording face sample for speaker: "
                            cv2.putText(img, info + name , (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                            cv2.imshow('camera',img) 
                            time.sleep(1)
                            names.clear()
                            with open('speakers.txt','r') as f:
                                while True:
                                    line = f.readline()
                                    if not line:
                                        break
                                    names.append(line.strip())
                            if name not in names:
                                speaker = Speaker(name)
                                actual_speaker = speaker
                                speakers[name] = speaker
                                # names.append(name)
                                # cam.release()
                                # cv2.destroyWindow('camera')
                                names.append(name)
                                face_registering_thread = Thread(target= register_new_face(name, gray, faces, img, cam, len(names)))
                                face_registering_thread.start()
                                face_registering_thread.join()
                                cv2.putText(img, info + name  , (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                cv2.imshow('camera',img)
                                time.sleep(2)
                                face_training_thread = Thread( target = train_algorithm )
                                face_training_thread.start()
                                face_training_thread.join()
                                # train_algorithm()
                                with open('speakers.txt', 'a') as f:
                                    f.write(name)
                            audio = Audio()
                            record_speaker_voice_sample_for_traning_thread = Thread( target =  audio.record_speaker_voice_sample_for_traning(name2codepoint, stream_segment))
                            record_speaker_voice_sample_for_traning_thread.start()
                k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    key = 'q'
                    print("\n [INFO] Exiting Program and cleanup stuff")
                    cam.release()
                    cv2.destroyAllWindows()
                    exit()
                if os.path.exists('./face_recognition/trainer/trainer.yml'): break
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('./face_recognition/trainer/trainer.yml')
        #iniciate id counter
        id = 0
        # names related to ids: example ==> Vladislav: id=1,  etc
        global hop
        hop = 0
        tmp_speaker = last_speaker
        while True:
            ret, img = cam.read()    
            img = cv2.flip(img, 1) # Flip vertically
            cv2.imshow('camera',img) 
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )
            face_recognized = False
            for(x,y,w,h) in faces:
                #cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                # If confidence is less than 100 ==> "0" : perfect match 
                if (confidence < 100):
                    while hop == 0:
                        print("hop")
                        tmp_speaker = last_speaker
                        hop += 1
                    id = names[id]
                    face_recognized = True
                    if id in speakers.keys():
                        actual_speaker.message_status = True
                        last_speaker = tmp_speaker
                        actual_speaker = speakers[id]
                        actual_speaker.message = message
                    else:
                        last_speaker = tmp_speaker
                        speaker = Speaker(id)
                        speakers[id] = speaker
                        actual_speaker = speaker
                        actual_speaker.message = message
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    last_speaker = actual_speaker
                    actual_speaker = Speaker(id)
                    actual_speaker.message_status = True
                    confidence = "  {0}%".format(round(100 - confidence))
                cv2.putText(img, actual_speaker.name , (x+7,y-5), font, 1, (0,255,0), 1)
                cv2.putText(img,str(confidence),(x+5,y+35),font,1,(255,255,0),0,5)   
            for each in name_subtring:
                if each in  message.strip() and actual_speaker.message_status:
                    index = actual_speaker.message.find(each)
                    tmp = actual_speaker.message[index+len(each):]
                    print("tmp>",tmp)
                    tmp_name = re.split(';. |, |\*|\n',tmp)[0]
                    name = tmp_name.split('.')[0].strip()
                    print("tmp>",name)
                    info = "Recording face sample for speaker: "
                    names.clear()
                    with open('speakers.txt','r') as f:
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            names.append(line.strip())
                    if name not in names:
                        print("NAME: " +  name)
                        # time.sleep(2)
                        speaker = Speaker(name)
                        # last_speaker = actual_speaker
                        actual_speaker = speaker
                        speakers[name] = speaker
                        audio = Audio()
                        audio.record_speaker_voice_sample_for_traning(name, stream_segment)
                        cam.release()
                        register_new_face(name ,gray, faces, img, cam, len(names))
                        names.append(name)
                        # face_registring_thread = Thread(target= register_new_face(name ,gray, faces, img, cam, len(names)))
                        # face_registring_thread.start()
                        # face_registring_thread.join()
                        train_algorithm()
                        cam = cv2.VideoCapture(0)
                        cv2.putText(img, info + name  , (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                        cv2.imshow('camera',img)
                        recognizer.read('./face_recognition/trainer/trainer.yml')
                        with open('speakers.txt', 'a') as f:
                            f.write( '\n' + name)
                        # last_message = actual_speaker.message
                        message = ""
                        actual_speaker.message = ""
                        actual_speaker.message_status = False
                        break
                        # continue
                    else: 
                        actual_speaker.message = ""
                        actual_speaker.message_status = False
                        break
                        
            if len(actual_speaker.message) > 40 and actual_speaker.message_status:
                message1 = actual_speaker.message[0:40]
                message2 = actual_speaker.message[40:]
                #self.frame = cv2.rectangle(self.frame, (10, 20), (620, 100),(blue, green, red), 2)
                if len(last_speaker.message) > 0 or (last_speaker.name != "Unknown") : 
                    cv2.putText(img,  last_speaker.name + ":" + last_speaker.message , (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_speaker.text_color, 2, cv2.LINE_AA)
                cv2.putText(img, actual_speaker.name +":" + message1, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, actual_speaker.text_color, 2, cv2.LINE_AA)
                cv2.putText(img, message2, (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, actual_speaker.text_color, 2, cv2.LINE_AA)
            elif len(actual_speaker.message) < 40 and len(actual_speaker.message) > 0 and actual_speaker.message_status:
                #self.frame = cv2.rectangle(img, (50, 20), (len(message)*15+100, 50),(blue, green, red), 2)
                if len(last_speaker.message) > 0 or (last_speaker.name != "Unknown") : 
                    cv2.putText(img,  last_speaker.name + ":" + last_speaker.message , (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_speaker.text_color, 2, cv2.LINE_AA)
                cv2.putText(img, actual_speaker.name +":" + actual_speaker.message, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, actual_speaker.text_color, 2, cv2.LINE_AA)
            # last_speaker.message = actual_speaker.message
            # if face_recognized :
            #     last_speaker = actual_speaker
            if (time.time() - message_start_time > 5 ) :
                actual_speaker.message_status = False
            if emergency != "":
                cv2.putText(img, "SMOKE ALARM!", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
                logo = cv2.imread('fire.jpg')
                size = 100
                logo = cv2.resize(logo, (size, size))
                img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                # Region of Interest (ROI), where we want
                # to insert logo
                roi = img[-size-10:-10, -size-10:-10]
                # Set an index of where the mask is
                roi[np.where(mask)] = 0
                roi += logo
                cv2.imshow('camera',img)   
            
            
            # cv2.putText(img, str(id), (x+7,y-5), font, 1, (255,255,255), 1)
        #    # cv2.putText(frame, message, (290,150), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_8)
                # cv2.putText(
                #             img, 
                #             str(confidence), 
                #             (x+5,y+h-5), 
                #             font, 
                #             1, 
                #             (255,255,0), 
                #             1
                #            )  
            cv2.imshow('camera',img) 
            k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                key = 'q'
                break
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
        exit()


import re
import sys
import time

from google.cloud import speech
import pyaudio
from six.moves import queue

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"


def get_current_time():
    """Return Current Time in MS."""
    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )
    def get_nr_of_channels(self):
        return self._num_channels
    def __enter__(self):

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):

        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""

        self._buff.put(in_data)
        return None, pyaudio.paContinue
    # chunk_nr = 0
    def generator(self):
        """Stream Audio from microphone to API and to local buffer"""
        global chunk_from 
        global chunk_to 
        global chunks_nr
        chunk_from = 0
        chunk_to = 0
        while not self.closed:
            data = []
            chunks_nr += 1 
            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )
                    chunk_from = chunks_from_ms
                    chunk_to = len(self.last_audio_input)
                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        # stream_segment.append(self.last_audio_input[i])
                        data.append(self.last_audio_input[i])

                self.new_stream = False
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)
            stream_segment.append(chunk)
            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)
def listen_print_loop(responses, stream, seconds):
    global chunk_from 
    global chunk_to 
    global chunks_nr
    global key
    global message	
    global message_start_time
    global actual_speaker
    global speakers
    global tmp_speaker
    global face_status
    global hop 
    global last_speaker
    audio = Audio()
    global a
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    for response in responses:
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        result_seconds = 0
        result_micros = 0
        # print("a="+str(a))
        # if str(a) == 10:
        #     last_result_seconds = 0
        #     last_result_micros = 0
        # else:
        #     print('DOINA')
        #     if last_transcript.result_end_time.seconds:
        #         last_result_seconds = last_transcript.result_end_time.seconds
        #     if last_transcript.result_end_time.microseconds:
        #         last_result_micros = last_transcript.result_end_time.microseconds 
        #     if result.result_end_time.seconds:
        #         result_seconds = result.result_end_time.seconds

        #     if result.result_end_time.microseconds:
        #         result_micros = result.result_end_time.microseconds
        #         stream.result_end_time = int((result_seconds * 1000) + int(result_micros / 1000))
        #         last_result_end_time = int((last_result_seconds * 1000) + int(last_result_micros / 1000))
        #         last_result_end_time_diff = int((result_seconds * 1000) + int(result_micros / 1000)) - int((last_result_seconds * 1000) + int(last_result_micros / 1000))
        #         print(" last_result_end_time" + str( last_result_end_time))
        #         print(" stream.result_end_time" + str( stream.result_end_time))
        #         print("last_result_end_time_diff" + str(last_result_end_time_diff))
        # last_transcript = transcript
        # a = a + 1
        
        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds   
        stream.result_end_time = int((result_seconds * 1000) + int(result_micros / 1000))
       
        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (STREAMING_LIMIT * stream.restart_counter)
        )
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:
            sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")
            message_start_time = time.time()	
            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True
            # last_message = last_speaker + ":" + actual_speaker.message
            speaker_name = actual_speaker.name
            msg = actual_speaker.message
            col = actual_speaker.text_color
            last_speaker = Speaker(speaker_name)
            last_speaker.message = msg
            last_speaker.text_color = col
            audio.record_speaker_voice_sample_for_verifying(actual_speaker, stream_segment[-chunks_nr:])
            chunks_nr = 0
            name = audio.verify_speaker_by_voice()
            if name in speakers.keys():
                actual_speaker = speakers[name]
            else:
                speaker  = Speaker(name)
                speakers[name] = speaker
                actual_speaker = speaker
            hop = 0
            actual_speaker.message = transcript
            message = transcript
            actual_speaker.message_status = True
            # print(len(stream_segment[-chunk_to]))
            # audio.record_speaker_voice_sample(actual_speaker, stream_segment[chunk_from:chunk_to])
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit|good bye)\b", transcript, re.I) or key == 'q':
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                key = 'q'
                stream.closed = True
                break
        else:
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")
            stream.last_transcript_was_final = False

def speech_recognition():
    global key
    """start bidirectional streaming from microphone input to speech API"""
    client = speech.SpeechClient()
    second_lang = "ro-Ro"
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US",
        # use_enhanced=True,
        # model="phone_call",
        # enable_automatic_punctuation=True,
        # alternative_language_codes=[second_lang],
        max_alternatives=1,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    with mic_manager as stream:

        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write(
                "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )

            stream.audio_input = []
            start = get_current_time()
            audio_generator = stream.generator()

            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )
            responses = client.streaming_recognize(streaming_config, requests)
            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream, get_current_time()-start )
            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1
            if not stream.last_transcript_was_final:
                sys.stdout.write("\n")
            stream.new_stream = True
message_start_time = time.time()

import pyaudio
from numpy import zeros,linspace,short,fromstring,hstack,transpose,log,frombuffer
from numpy.fft import fft
from time import sleep
def smoke_alarm_detection():
    global key
    time.sleep(5)
    #Volume Sensitivity, 0.05: Extremely Sensitive, may give false alarms
    #             0.1: Probably Ideal volume
    #             1: Poorly sensitive, will only go off for relatively loud
    SENSITIVITY= 0.1
    # Alarm frequencies (Hz) to detect (Use audacity to record a wave and then do Analyze->Plot Spectrum)
    TONE = 3500
    #Bandwidth for detection (i.e., detect frequencies within this margin of error of the TONE)
    BANDWIDTH = 30
    #How many 46ms blips before we declare a beep? (Take the beep length in ms, divide by 46ms, subtract a bit)
    beeplength=8
    # How many beeps before we declare an alarm?
    alarmlength=5
    # How many false 46ms blips before we declare the alarm is not ringing
    resetlength=10
    # How many reset counts until we clear an active alarm?
    clearlength=30
    # Enable blip, beep, and reset debug output
    debug=False
    # Show the most intense frequency detected (useful for configuration)
    frequencyoutput=True
    global emergency
    device_index = 2
    #Set up audio sampler - 
    NUM_SAMPLES = 2048
    SAMPLING_RATE = 44100
    pa = pyaudio.PyAudio()
    _stream = pa.open(format=pyaudio.paInt16,
                      channels=1, rate=SAMPLING_RATE,
                      input=True, input_device_index = 1,
                      frames_per_buffer=NUM_SAMPLES)
    
    print("Alarm detector working. Press CTRL-C to quit.")
    blipcount=0
    beepcount=0
    resetcount=0
    clearcount=0
    alarm=False
    
    while True:
        while _stream.get_read_available()< NUM_SAMPLES:
                sleep(0.01)
        audio_data  = frombuffer(_stream.read(
             _stream.get_read_available()), dtype=short)[-NUM_SAMPLES:]
        # Each data point is a signed 16 bit number, so we can normalize by dividing 32*1024
        normalized_data = audio_data / 32768.0
        intensity = abs(fft(normalized_data))[:NUM_SAMPLES//2]
        frequencies = linspace(0.0, float(SAMPLING_RATE)/2, num=NUM_SAMPLES//2)
        if frequencyoutput:
            which = intensity[1:].argmax()+1
            # use quadratic interpolation around the max
            if which != len(intensity)-1:
                y0,y1,y2 = log(intensity[which-1:which+2:])
                x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
                # find the frequency and output it
                thefreq = (which+x1)*SAMPLING_RATE/NUM_SAMPLES
            else:
                thefreq = which*SAMPLING_RATE/NUM_SAMPLES
            #print( "\t\t\t\tfreq=",thefreq)
        if max(intensity[(frequencies < TONE+BANDWIDTH) & (frequencies > TONE-BANDWIDTH )]) > max(intensity[(frequencies < TONE-1000) & (frequencies > TONE-2000)]) + SENSITIVITY:
            blipcount+=1
            resetcount=0
            if debug: print ("\t\tBlip",blipcount)
            if (blipcount>=beeplength):
                blipcount=0
                resetcount=0
                beepcount+=1
                if debug: print( "\tBeep",beepcount)
                if (beepcount>=alarmlength):
                    clearcount=0
                    alarm=True
                    print ("Alarm!")
                    emergency = "smoke"
                    beepcount=0
        else:
            blipcount=0
            resetcount+=1
            if debug: print( "\t\t\treset",resetcount)
            if (resetcount>=resetlength):
                resetcount=0
                beepcount=0
                if alarm:
                    clearcount+=1
                    if debug: print ("\t\tclear",clearcount)
                    if clearcount>=clearlength:
                        clearcount=0
                        print ("Cleared alarm!")
                        emergency = ""
                        alarm=False
        if key == 'q':
            break
        sleep(0.01)
    _stream.stop_stream()
    _stream.close()
    pa.terminate()
        
if __name__ == "__main__":
    global actual_speaker
    global names
    global speakers
    global emergency
    global info
    global face_status
    face_status = False
    global _audio
    global chunks_nr
    chunks_nr = 0
    global a
    a = 0
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= "C:\\Users\\Grejd\\OneDrive\\Рабочий стол\\TEZA\\Optical_Whisper_venv\\Optical_Whisper\\source\\google_key\\optical-whisper-334216-99b674934e13.json"
    _audio = Audio()
    emergency = ""
    actual_speaker = Speaker('Unknown')
    last_speaker = actual_speaker
    speakers = {"Unknown": actual_speaker}
    names = []
    with open('speakers.txt','r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            names.append(line.strip())
    video = Video()
    video_thread = Thread(target = video.recording_video)
    video_thread.start()
    audio_thread = Thread(target = speech_recognition)
    audio_thread.start()
    alarm_thread = Thread(target = smoke_alarm_detection)
    alarm_thread.start()

    video_thread.join()
    print("video end")
    audio_thread.join()
    print("audio end")
    alarm_thread.join()
    print("alarm end")