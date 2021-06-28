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
import pandas as pd


def visualization(times,vis=0, disengaged_student=0,output_name ="/Engaged", labels="", savefolder="result",engaged_student=0,highly_engaged_student=0,sizes="",student_ID=""):
    '''
    times: the vertical bar
    vis: type of visualize: 0: student engaged, 1: student disengaged, 2: student highly engaged, 3: stackplot for 3 type of students
    engaged_student: number of engaged student
    
    labels: include 3 type of student we need to show'''
    fig = plt.figure()
    width = 0.2  
    if vis == 0:
        plt.xlabel('Times(sec)') 
        plt.ylabel('Number of '+output_name[1:] +" Students") 
        plt.title('Engagement Visualization')
        plt.plot(times,disengaged_student)
        #plt.ylim(0, 11)
        plt.savefig(savefolder + output_name + ".png")
        plt.clf()
        
    elif vis == 1: 
    	fig, ax = plt.subplots()
    	ax.stackplot(times, disengaged_student,engaged_student,highly_engaged_student, labels=labels)
    	ax.set_title('Engagement Visualization')
    	ax.set_xlabel('Times(sec)') 
    	ax.set_ylabel('Number of '+output_name[1:] +" Students")
    	ax.legend(labels=labels)
    	plt.ylim(0, 11)
    	plt.savefig(savefolder + "/engagement.png")
    	
    	plt.clf()
    elif vis == 2:
    	fig, ax = plt.subplots()
    	ax.bar(times, disengaged_student, width, color='r')
    	ax.bar(times, engaged_student, width,bottom=disengaged_student, color='b')
    	ax.bar(times, highly_engaged_student, width,bottom=engaged_student, color='g')
    	ax.set_xlabel('Time (sec)')
    	ax.set_ylabel('Number of student')
    	ax.set_title('Engagement Visualization')
    	ax.legend(labels=labels)
    	plt.ylim(0, 11)
    	plt.savefig(savefolder +"/combine.png")


def data_visualization(filename):
	data = pd.read_csv(filename,index_col=0)
	df = pd.DataFrame(data)
	
	# list of student name
	student_name=df.groupby("Student_ID").size().index
	
	
	# list of engagement level
	Engagement_Level = df.groupby("Engagement status").size()
	
	
	###################sumarization for engagement detection
	fig = plt.figure()
	s = df.groupby(["frame","Engagement status"]).size().unstack().fillna(0)
	s=pd.DataFrame(s,columns=s.columns)
	print(s)
	fig = s.plot.area(xticks=range(0,180,20),yticks=range(0,10,1), ylabel="Number of Students", xlabel="Times(sec)", title="Sumarization of Engagement Detection").get_figure()
	fig.savefig("result/sumarization.jpg")
	
	###################visualize for engagement Level:
	for i in Engagement_Level.index:
		fig = plt.figure()
		
		#sort by each engagement level: disengaged, engaged, highly-engaged
		engagement_stt = df[df["Engagement status"]==i]
		student= pd.Series(engagement_stt.groupby("frame").size(),name="Engagement Visualization" )
		#print(student)
		fig = student.plot(subplots=True, figsize=(5, 8),ylabel = "Number of "+ i +" students",xlabel="Times(sec)", title="Engagement Visualization")[0].get_figure()
		fig.savefig("result/test"+i+".jpg")
		
	####################visualize the information for each student: pie chart
	for i in student_name:
		fig = plt.figure()
    		
		df1 = df.loc[i,:]
		df2 = pd.Series(df1.groupby("Engagement status").size(),name="visualization for "+i)
		fig = df2.plot.pie(subplots=True, figsize=(4, 4),autopct="%.2f%%",labels=None)[0].get_figure()
		fig.legend(title = "Engagement Level:",labels=df2.index)
		fig.savefig("result/"+i+".jpg")

def engagement_detection(file_name):

	# Initializing
	print(os.getcwd() + file_name)
	cap = cv2.VideoCapture(os.getcwd() + file_name)
	#Get the Default resolutions
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	fps = cap.get(cv2.CAP_PROP_FPS)
	print(fps, cap.get(cv2.CAP_PROP_FPS))
	# Define the codec and filename.
	out = cv2.VideoWriter(os.getcwd()+'/result'+ file_name[6:-4] + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
	print(os.getcwd()+'/result'+ file_name[6:-4])
	disengaged_student = []
	engaged_student = []
	highly_engaged_student = []
	video_times = []
	labels= ["Disengaged Students", "Engaged Students","Highly_engaged Students"]
	info_of_each_st = {}
	#encode database for face recognition
	encodeFaceList,classNames =face_recog.encode_face()

	#cap = cv2.VideoCapture(0)
	ana = analysis()


	#CSV file initializing :

	filename = "result"+file_name[6:-4] + ".csv"
	row_contents = ['Student_ID','Emotion','Engagement status', 'frame']

	with open(filename, 'w', newline='') as outcsv:
		writer = csv.writer(outcsv)
		writer.writerow(['Student_ID','Emotion','Engagement status', 'frame'])
	       
	# Capture every frame and send to detector
		while(cap.isOpened()):
		    ret, frame = cap.read()
		    if ret == True:
		        #detect emotion and engagement for every frame
		    	bm, engaged_students,info_std = ana.detect_face(encodeFaceList,classNames,frame)
		    	
		    	highly_engaged_student.append(engaged_students[0])
		    	engaged_student.append(engaged_students[1])
		    	disengaged_student.append(engaged_students[2])
		    	
		    	# write the frame
		    	out.write(frame)
		    	cv2.imshow("Frame", frame)
		    	time = round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000,2)
		    	video_times.append(time)
		    	print(cap.get(cv2.CAP_PROP_POS_MSEC))
		    	#visualize for each student
		    	#write csv file
		    	for i in info_std:
		    	    # info of each student: each student will be (student ID, emotion, engaged status,time)
		    	    writer.writerow([i[0],i[1],i[2],time])
		    	
		    	# Exit if 'q' is pressed
		    	if cv2.waitKey(1) & 0xFF == ord('q'):
		    	    break
		    else:
		    	break

	#visualize the result
	visualization(video_times, 0, engaged_student)
	visualization(video_times, 0, disengaged_student,"/Disengaged")
	visualization(video_times, 0, highly_engaged_student,"/Highly-engaged")
	visualization(video_times, 1, disengaged_student,engaged_student=engaged_student, highly_engaged_student = highly_engaged_student, labels=labels)
	visualization(video_times, 2, disengaged_student,engaged_student=engaged_student, highly_engaged_student = highly_engaged_student, labels=labels)



	# Release the memory
	cap.release()
	out.release()
	cv2.destroyAllWindows()          


	#visualize the result

#engagement_detection('/input/123CMNR.mp4')
data_visualization("result/123CMNR.csv")
