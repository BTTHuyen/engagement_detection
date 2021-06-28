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
	fig = s.plot.area(xticks=range(0,460,50),yticks=range(0,11,1), ylabel="Number of Students", xlabel="Times(sec)", title="Sumarization of Engagement Detection").get_figure()
	fig.savefig("result/sumarization.jpg")
	
	###################visualize for engagement Level:
	for i in Engagement_Level.index:
		fig = plt.figure()
		
		#sort by each engagement level: disengaged, engaged, highly-engaged
		engagement_stt = df[df["Engagement status"]==i]
		student= pd.Series(engagement_stt.groupby("frame").size(),name="Engagement Visualization" )
		fig = student.plot(subplots=True, figsize=(5, 8),ylabel = "Number of "+ i +" students",xlabel="Times(sec)", title="Engagement Visualization")[0].get_figure()
		fig.savefig("result/test"+i+".jpg")
		
	####################visualize the information for each student: pie chart
	for i in student_name:
		print(i)
		fig = plt.figure()
    		
		df1 = df.loc[i,:]
		df2 = pd.Series(df1.groupby("Engagement status").size(),name="visualization for "+str(i))
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



	# Release the memory
	cap.release()
	out.release()
	cv2.destroyAllWindows()          


	#visualize the result

engagement_detection('/input/video_2021_3_11_multiface.mp4')
#data_visualization("result/123CMNR.csv")
