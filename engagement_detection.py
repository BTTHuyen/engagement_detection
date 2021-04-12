# Run this file on CV2 in local machine to construct a Concentration Index (CI).
# Video image will show emotion on first line, and engagement on second. Engagement/concentration classification displays either 'Pay attention', 'You are engaged' and 'you are highly engaged' based on CI. Webcam is required.
# Analysis is in 'Util' folder.


from analysis_realtime import analysis
import cv2
import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import face_recog
import time


def visualization(times,vis=0, student_engaged=0, student_disengaged=0, student_highly_engaged=0, labels="", savefolder="result"):
    '''
    times: the vertical bar
    vis: type of visualize: 0: student engaged, 1: student disengaged, 2: student highly engaged, 3: stackplot for 3 type of students
    student_engaged: number of student engaged
    
    labels: include 3 type of student we need to show'''
    if vis == 0:
        plt.xlabel('times') 
        plt.ylabel('number of student engaged') 
        plt.title('display')
        plt.plot(times,student_engaged)
        plt.savefig(savefolder + "/engaged.png")
        plt.clf()
    elif vis == 1:
        plt.xlabel('times') 
        plt.ylabel('number of student disengaged') 
        plt.title('display')
        plt.plot(times,student_disengaged)
        plt.savefig(savefolder + "/disengaged.png")
        plt.clf()
    elif vis == 2:
        plt.xlabel('times') 
        plt.ylabel('number of student highly engaged') 
        plt.title('display')
        plt.plot(times,student_highly_engaged)
        plt.savefig(savefolder + "/highly_engaged.png")
        plt.clf()
    elif vis == 3: 
        plt.stackplot(times, student_disengaged,student_engaged,student_highly_engaged)
        plt.savefig(savefolder + "/area_chart.png")
        plt.clf()
    #plt.show()




#CSV file initializing :

filename = "f.csv"
row_contents = ['x','y','time']

with open(filename, 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["y", "x","time"])


# Initializing
cap = cv2.VideoCapture(os.getcwd() + '/input/video_2021_3_11_multiface.mp4')
#Get the Default resolutions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.set(cv2.CAP_PROP_FPS, 12)
print(fps, cap.get(cv2.CAP_PROP_FPS))
# Define the codec and filename.
out = cv2.VideoWriter(os.getcwd()+'/result/output_multiface.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

student_disengaged = []
student_engaged = []
student_highly_engaged = []
video_times = []
labels= ["student_disengaged", "student_engaged","student_highly_engaged"]
encodeFaceList,classNames =face_recog.encode_face()

#cap = cv2.VideoCapture(0)
ana = analysis()

# Capture every frame and send to detector
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        bm, disengaged, engaged, highly_engaged = ana.detect_face(encodeFaceList,classNames,frame)
        student_disengaged.append(disengaged)
        student_engaged.append(engaged)
        student_highly_engaged.append(highly_engaged)
        # write the frame
        out.write(frame)
        cv2.imshow("Frame", frame)
        video_times.append(round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000,2))
        #print("video time: ",video_times)
        print(cap.get(cv2.CAP_PROP_POS_MSEC))
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#visualize the result
visualization(video_times, 0, student_engaged)
visualization(video_times, 1, student_disengaged = student_disengaged)
visualization(video_times, 2, student_highly_engaged = student_highly_engaged)
visualization(video_times, 3, student_disengaged,student_engaged, student_highly_engaged,labels=labels)

# Release the memory

cap.release()
out.release()
cv2.destroyAllWindows()          


#visualize the result



