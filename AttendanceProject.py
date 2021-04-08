import cv2 
import numpy as np 
import face_recognition
import os
from datetime import datetime
import csv
#LOAD IMAGE AND CONVERT TO RGB IMAGE

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        print(img.shape)
        img= cv2.resize(img, (512,440), interpolation = cv2.INTER_AREA)
        #print(img.shape)
        encode = face_recognition.face_encodings(img)[0]
        #print(encode)
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        # print(myDataList)  
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            print(nameList)
        if name not in  nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'/{name},{dtString}')



path = 'students'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

encodeListKnown = findEncoding(images)
print('Encoding Complete, the total images is:',len(encodeListKnown))
cap = cv2.VideoCapture("input/video_2021_3_11_multiface.mp4")

while True:
    success, img = cap.read()
    #imgS = cv2.resize(img,(0,0), None, 0.6,0.6)
    imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceLoc)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

            y1,x2,y2,x1 = faceLoc
            #y1,x2,y2,x1 = y1,x2,y2,x1 
            cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0),2)
            cv2.rectangle(img,(x1,y2-35), (x2,y2), (0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            #markAttendance(name)



    cv2.imshow('Webcam',img)
    cv2.waitKey(1)