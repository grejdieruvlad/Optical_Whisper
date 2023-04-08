


class Video:
        
    def record_video_stream():
    
        print()
    def take_speaker_shots():
        #cam = cv2.VideoCapture(0)
        #cam.set(3, 1920) # set video width
        #cam.set(4, 1080) # set video height
        face_detector = cv2.CascadeClassifier('../face_recognition/haarcascade_frontalface_default.xml')
        # For each person, enter one numeric face id
        #face_id = input('\n enter user id end press <return> ==>  ')
        face_id = len(speakers)+1
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
                cv2.imwrite("../face_recognition/dataset/User." + str(face_id) + '.' +  
                            str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('image', img)
            k = cv2.waitKey(1) # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                break
        # Do a bit of cleanup
        #print("\n [INFO] Exiting Program and cleanup stuff")
        #cam.release()
        #cv2.destroyAllWindows()
    def train_face_recognition_model():
        
        print()

    def verify_speaker_by_face():
        print()
    
    def show_video_stream_with_AR():
        print()