# Program constructs Concentration Index and returns a classification of engagement.

import cv2
import numpy as np
import dlib
from math import hypot
from keras.models import load_model
import csv
import time
import face_recog
from datetime import datetime



filename = "f.csv"

temp = 0



class analysis:

    # Initialise models
    def __init__(self):
        self.emotion_model = load_model('./util/model/emotion_recognition.h5')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "./util/model/shape_predictor_68_face_landmarks.dat")
        self.faceCascade = cv2.CascadeClassifier(
            './util/model/haarcascade_frontalface_default.xml')
        self.x = 0
        self.y = 0
        self.emotion = 5
        self.size = 0
        self.frame_count = 0
        self.studentname =""
        
# Function for finding midpoint of 2 points
    def midpoint(self, p1, p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# Function for eye size
    def get_blinking_ratio(self, frame, eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(
            eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(
            eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.midpoint(facial_landmarks.part(
            eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(
            eye_points[5]), facial_landmarks.part(eye_points[4]))
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        hor_line_lenght = hypot(
            (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot(
            (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        ratio = ver_line_lenght / hor_line_lenght
        return ratio

  # Gaze detection function
    def get_gaze_ratio(self, frame, eye_points, facial_landmarks, gray):

        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(
                                        eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(
                                        eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x,
                                     facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x,
                                     facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        up_side_threshold = threshold_eye[0: int(height/2), 0: int(width / 2)]
        up_side_white = cv2.countNonZero(up_side_threshold)
        down_side_threshold = threshold_eye[int(height/2): height, 0: width]
        down_side_white = cv2.countNonZero(down_side_threshold)
        lr_gaze_ratio = (left_side_white+10) / (right_side_white+10)
        ud_gaze_ratio = (up_side_white+10) / (down_side_white+10)
        return lr_gaze_ratio, ud_gaze_ratio

# -------------------------------------Main function for analysis----------------------------------

    def detect_face(self, encodeFaceList,classNames, frame):
        global temp
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #faces = self.detector(gray)
        
        ######Face recognition usinng opencv to determine whose face are they
        faces=face_recog.face_detection(encodeFaceList,classNames,frame=gray)
        
        benchmark = []
        
        emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                            3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
                            
        engagements = {0:'highly engaged', 1: 'engaged', 2:'disengaged'}
        
        color = {0:(255, 255, 0), 1: (0, 255, 0), 2:(0, 0, 255)}
        
        #number of disengaged, engagedn highly engage students on each frame
        engaged_student = [0]*3 # highly_engaged, engaged, disengaged

        stt = 0    # stt=0: highly engaged student, stt = 1: engaged student, stt=2: disengaged stduent
        
        # info of each student: each student will be (student ID, emotion, engaged status)
        info_std=[]
        if not faces:
            temp = temp + 1
            v1 = str(0)
            v2 = str(temp)
            with open(filename,'a+', newline='') as csvfile:
                csvwrite = csv.writer(csvfile)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                csvwrite.writerows([[v1,v2,current_time]])
        else:
            for face in faces:
                #x, y = face.left(), face.top()
                #x1, y1 = face.right(), face.bottom()
                
                #change to format of dlib face recognition
                x,y,x1,y1,studentName = face
                self.studentname = studentName
                print(studentName)
                face = dlib.rectangle(x,y,x1,y1)
                
                
                
                f = gray[x:x1, y:y1]
                #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                landmarks = self.predictor(gray,face)
                left_point = (landmarks.part(36).x, landmarks.part(36).y)
                right_point = (landmarks.part(39).x, landmarks.part(39).y)
                center_top = self.midpoint(landmarks.part(37), landmarks.part(38))
                center_bottom = self.midpoint(
                    landmarks.part(41), landmarks.part(40))
                hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
                ver_line = cv2.line(frame, center_top,
                                    center_bottom, (0, 255, 0), 2)



            #Exp
                left_point = (landmarks.part(42).x, landmarks.part(42).y)
                right_point = (landmarks.part(45).x, landmarks.part(45).y)
                center_top = self.midpoint(landmarks.part(43), landmarks.part(44))
                center_bottom = self.midpoint(
                    landmarks.part(47), landmarks.part(46))
                hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
                ver_line = cv2.line(frame, center_top,
                                    center_bottom, (0, 255, 0), 2)


                right_eye_ratio = self.get_blinking_ratio(frame,
                                                        [42, 43, 44, 45, 46, 47], landmarks)

                right_gaze_ratio_lr, right_gaze_ratio_ud = self.get_gaze_ratio(frame,
                                                                [42, 43, 44, 45, 46, 47], landmarks, gray)

            #Exp end


                left_eye_ratio = self.get_blinking_ratio(frame,
                                                        [36, 37, 38, 39, 40, 41], landmarks)

                left_gaze_ratio_lr, left_gaze_ratio_ud = self.get_gaze_ratio(frame,
                                                                [36, 37, 38, 39, 40, 41], landmarks, gray)

            #Exp
                eye_ratio = max(right_eye_ratio , left_eye_ratio)
                gaze_ratio_lr = max(left_gaze_ratio_lr , right_gaze_ratio_lr)
                gaze_ratio_ud = max(left_gaze_ratio_ud , right_gaze_ratio_ud)
                
                self.x = gaze_ratio_lr
                self.y = gaze_ratio_ud
                self.size = eye_ratio
            #end exp

                benchmark.append([gaze_ratio_lr, gaze_ratio_ud, eye_ratio])
                emotion = self.detect_emotion(gray)

		#determine engaged student based on concentration index
                ci = self.gen_concentration_index()     
                
                if ci > 0.6:
                    engaged_student[0] +=1
                    stt = 0
                elif ci > 0.3 and ci <=0.6:
                    engaged_student[1] += 1
                    stt = 1
                else:
                    engaged_student[2] +=1
                    stt = 2
                    
               
                #information of each student
                info_std.append([self.studentname, emotions[self.emotion], engagements[stt]])
                     
                #draw label
                label = self.studentname + "-" + emotions[self.emotion] + "_" + engagements[stt]
                labelSize=cv2.getTextSize(label,font,0.5,2)
               
               
                _x1 = x
                _y1 = y+5
                _x2 = x+labelSize[0][0] + 5
                _y2 = y-int(labelSize[0][1]) - 5
                cv2.rectangle(frame,(_x1,_y1),(_x2,_y2),color[stt],cv2.FILLED)
                cv2.rectangle(frame, (x, y), (x1, y1), color[stt], 2)
                cv2.putText(frame,label, (x, y), font, 0.5, (0,0,0), 1)
                
        #print(info_std)        
        return frame,engaged_student, info_std

# -----------------------Function for detecting emotion-----------------------------------------

    def detect_emotion(self, gray):
        global temp
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Dictionary for emotion recognition model output and emotions
        emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                    3: 'Sad', 4: 'Surprised', 5: 'Neutral'}

        # Face detection takes approx 0.07 seconds
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100),)
        if len(faces) > 0:
            for x, y, width, height in faces:
                cropped_face = gray[y:y + height, x:x + width]
                test_image = cv2.resize(cropped_face, (48, 48))
                test_image = test_image.reshape([-1, 48, 48, 1])

                test_image = np.multiply(test_image, 1.0 / 255.0)

                # Probablities of all classes
                # Finding class probability takes approx 0.05 seconds
                if self.frame_count % 20 == 0:
                    probab = self.emotion_model.predict(test_image)[0] * 100
                    #print("--- %s seconds ---" % (time.time() - start_time))
                    # Finding label from  probabilities
                    # Class having highest probability considered output label
                    label = np.argmax(probab)
                    probab_predicted = int(probab[label])
                    predicted_emotion = emotions[label]
                    self.frame_count = 0
                    self.emotion = label

        else:

            temp = temp + 1
            v1 = str(0)
            v2 = str(temp)
            with open(filename,'a+', newline='') as csvfile:
                csvwrite = csv.writer(csvfile)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                csvwrite.writerows([[v1,v2,current_time]])
      

        self.frame_count += 1

        # # 	Weights from Sharma et.al. (2019)
        # Neutral	0.9
        # Happy 	0.6
        # Surprised	0.6
        # Sad	    0.3

        # Anger	    0.25
        # Fearful	0.3
        # 0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprised', 5: 'Neutral'}


    #----------------------------------concentration function ------------------------------
    def gen_concentration_index(self):
        global temp 
        weight = 0
        emotionweights = {0: 0.25, 1: 0.3, 2: 0.6,
                          3: 0.3, 4: 0.6, 5: 0.9}


        # 	      Open Semi Close
        # Centre	5	1.5	0
        # Upright	2	1.5	0
        # Upleft	2	1.5	0
        # Right	    2	1.5	0
        # Left	    2	1.5	0
        # Downright	2	1.5	0
        # Downleft	2	1.5	0
        gaze_weights = 0

        '''if self.size < 0.24:
            gaze_weights = 0
        elif self.size >= 0.24 and self.size < 0.28:
            gaze_weights = 1.5
        else:
            if self.x ==1.0 and self.y == 1.0:
                gaze_weights = 5
            elif (self.x < 2 and self.x > 1) and (self.y < 2 and self.y > 1) :
                gaze_weights = 4
            else:
                gaze_weights = 2 '''

        if self.size < 0.2:
            gaze_weights = 0
        elif self.size > 0.2 and self.size < 0.3:
            gaze_weights = 1.5
        else:
            if self.x < 2 and self.x > 1:
                gaze_weights = 5
            else:
                gaze_weights = 2

# Concentration index is a percentage : max weights product = 4.5
        concentration_index = (
            emotionweights[self.emotion] * gaze_weights) / 4.5
        #print(concentration_index,self.emotion,gaze_weights )
        temp = temp + 1
        v1 = str(concentration_index)
        v2 = str(temp)

        with open(filename,'a+', newline='') as csvfile:
            csvwrite = csv.writer(csvfile)
            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")

            csvwrite.writerows([[v1,v2,current_time]])
        return concentration_index
