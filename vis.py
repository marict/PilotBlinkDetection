import pandas as pd
import matplotlib.pyplot as plt
import scipy
import argparse
import imutils
import time
import dlib
import cv2
import sys,os
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from operator import itemgetter
from numpy import nan
from sklearn.preprocessing import *

# Every variable declared here is a global
# basePath = "C:\\Users\\Paul\\Desktop\\Research\\PilotBlinkDetection\\"
basePath = os.path.dirname(os.path.realpath(__file__)) + "\\"

vidPath = basePath + "vids\\"
csvPath = basePath + "logs\\"
picPath = basePath + "pics\\"
preTrainedPath = basePath + "pre_trained_models\\"

detector2Path = preTrainedPath + "haarcascade_frontalface_default.xml"
detector3Path = preTrainedPath + "mmod_human_face_detector.dat"
shapePredPath = preTrainedPath + "shape_predictor_68_face_landmarks.dat"

# returns: 
# rect = dlib.rectangle or None if no faces found
# detectType to say which detector the rectangle represents
#
# if detector does not find a face than detector2 will try
def detectFaces(gray,FACE_DOWNSAMPLE_RATIO):
	detectType = 1
	
	# rescale frame for face detection
	gray_rescale = cv2.resize(gray, (0,0),fx=1.0/FACE_DOWNSAMPLE_RATIO,fy=1.0/FACE_DOWNSAMPLE_RATIO)
	
	# detect faces in the grayscale frame
	rects = detector(gray_rescale, 0)

	# dets = detector3(gray_rescale,0)
	
	# rects = dlib.rectangles()
	# rects.extend([d.rect for d in dets])
	
	# note rect is a dlib rectangle
	if(len(rects) == 0):
		# if no face found, try second detector
		faces = detector2.detectMultiScale(gray_rescale,1.3,5)
		if(len(faces) == 0):
			return (None,None)
	
		faces = sorted(faces,reverse = True,key=lambda x: x[2] * x[3])
		# use largest face found
		face = faces[0]
		(x,y,w,h) = face
		rect = dlib.rectangle(left=(x+w).item(), bottom=(y+h).item(), right=x.item(), top=y.item())
		detectType = 2
	else:
	
		# use largest face found
		# use first face found
		rects = sorted(rects,reverse=True,key=lambda x: x.area())
		rect = rects[0]
		
	return rect, detectType

# flip frame
def horizontal_flip(src):
	return cv2.flip(src,-1)
	

def distance_nose_cheek(shape):
	distance = dist.euclidean(shape[3],shape[33])
	print(distance)
	return distance

# Generates a csv string of the ears of each frame (nothing if no face detected)
def show_landmarks(filename, SHOW_FRAME = True, FLIP = False):
	print("Generating landmarks for " + filename)
	
	# How much to downsample face for detection
	FACE_DOWNSAMPLE_RATIO = 4

	# output matrix
	out = []
	
	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively

	# start the video stream thread
	vs = cv2.VideoCapture(filename)

	time.sleep(1.0)
	FRAME_NUM = 0
	print("Applying classifer to " + filename)
	
	print("Video properties:")
	FPS = vs.get(cv2.CAP_PROP_FPS)
	FC = vs.get(cv2.CAP_PROP_FRAME_COUNT)
	F = vs.get(cv2.CAP_PROP_FORMAT)
	W = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
	H = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
	print("\tFPS = " + str(FPS))
	print("\tFrame count = " + str(FC))
	print("\tFormat = " + str(F))
	print("\t(width,height) = (" + str(W) + "," + str(H) + ")")
	
	# "dropped" frames
	NOT_GRABBED = 0
	
	# loop over frames from the video stream
	# stop when we haven't grabbed END_VIDEO_LIMIT frames
	while FRAME_NUM < FC:
	
		# Try to grab frame
		(grabbed,frame) = vs.read()
		
		if(grabbed):
			FRAME_NUM += 1
			NOT_GRABBED = 0
		else:
			NOT_GRABBED += 1
			continue
			
		if(FLIP):
			frame = horizontal_flip(frame)
	
		# output timestamp and frame
		#timestamp = 1000.0 * float(FRAME_NUM)/FPS
		timestamp = FRAME_NUM
		
		# Apply transformations (one of them being grayscale)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		rect, detectType = detectFaces(gray, FACE_DOWNSAMPLE_RATIO)
		
		if(rect is not None):
			# Resize obtained rectangle for full resolution image
			rect_resize = dlib.rectangle(
				left=rect.left() * FACE_DOWNSAMPLE_RATIO, 
				top=rect.top() * FACE_DOWNSAMPLE_RATIO, 
				right=rect.right() * FACE_DOWNSAMPLE_RATIO, 
				bottom=rect.bottom() * FACE_DOWNSAMPLE_RATIO
			)	
		
			shape = predictor(gray,rect_resize)
			shape = face_utils.shape_to_np(shape)
			
			# draw landmarks on face
			distance_nose_cheek(shape)
			for (x, y) in [shape[3]]+[shape[33]]:
				cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
				
		if(SHOW_FRAME):
			display_frame = cv2.resize(frame,(720,480))
			if(rect is None):
				cv2.putText(display_frame, "NO FACE DETECTED", (300, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				print("nothing")
				
			cv2.putText(display_frame, "Frame: {}".format(FRAME_NUM), (10, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
			# show the frame
			cv2.imshow("Frame", display_frame)
			
		# output different csv things if face was detected
		if(rect is not None):
			out.append([int(timestamp),nan])
		else:
			out.append([int(timestamp),nan])
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			return out
			break
				
	# do a bit of cleanup
	cv2.destroyAllWindows()
	
	return np.asarray(out)
	
# Generates a csv string of the ears of each frame (nothing if no face detected)
def show_landmarks_webcam(SHOW_FRAME = True, FLIP = False):
	print("Generating landmarks for webcam")
	
	# How much to downsample face for detection
	FACE_DOWNSAMPLE_RATIO = 4

	# output matrix
	out = []
	
	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively

	# start the video stream thread
	vs = VideoStream(src=0).start()
	
	time.sleep(1.0)
	FRAME_NUM = 0
	
	print("Video properties:")
	FPS = vs.get(cv2.CAP_PROP_FPS)
	FC = vs.get(cv2.CAP_PROP_FRAME_COUNT)
	F = vs.get(cv2.CAP_PROP_FORMAT)
	W = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
	H = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
	print("\tFPS = " + str(FPS))
	print("\tFrame count = " + str(FC))
	print("\tFormat = " + str(F))
	print("\t(width,height) = (" + str(W) + "," + str(H) + ")")
	
	# "dropped" frames
	NOT_GRABBED = 0
	
	# loop over frames from the video stream
	# stop when we haven't grabbed END_VIDEO_LIMIT frames
	while FRAME_NUM < FC:
	
		# Try to grab frame
		(grabbed,frame) = vs.read()
		
		if(grabbed):
			FRAME_NUM += 1
			NOT_GRABBED = 0
		else:
			NOT_GRABBED += 1
			continue
			
		if(FLIP):
			frame = horizontal_flip(frame)
	
		# output timestamp and frame
		#timestamp = 1000.0 * float(FRAME_NUM)/FPS
		timestamp = FRAME_NUM
		
		# Apply transformations (one of them being grayscale)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		rect, detectType = detectFaces(gray, FACE_DOWNSAMPLE_RATIO)
		
		if(rect is not None):
			# Resize obtained rectangle for full resolution image
			rect_resize = dlib.rectangle(
				left=rect.left() * FACE_DOWNSAMPLE_RATIO, 
				top=rect.top() * FACE_DOWNSAMPLE_RATIO, 
				right=rect.right() * FACE_DOWNSAMPLE_RATIO, 
				bottom=rect.bottom() * FACE_DOWNSAMPLE_RATIO
			)	
		
			shape = predictor(gray,rect_resize)
			shape = face_utils.shape_to_np(shape)
			
			# draw landmarks on face
			distance_nose_cheek(shape)
			for (x, y) in [shape[3]]+[shape[33]]:
				cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
				
		if(SHOW_FRAME):
			display_frame = cv2.resize(frame,(720,480))
			if(rect is None):
				cv2.putText(display_frame, "NO FACE DETECTED", (300, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				print("nothing")
				
			cv2.putText(display_frame, "Frame: {}".format(FRAME_NUM), (10, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
			# show the frame
			cv2.imshow("Frame", display_frame)
			
		# output different csv things if face was detected
		if(rect is not None):
			out.append([int(timestamp),nan])
		else:
			out.append([int(timestamp),nan])
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			return out
			break
				
	# do a bit of cleanup
	cv2.destroyAllWindows()
	
	return np.asarray(out)
# ------------------------------------------------------------------------------------------------
	
# Data visualization file
print("[INFO] loading facial classifiers...")
# dlibs HOG based classifier
detector = dlib.get_frontal_face_detector()
# openCVs HAAR cascade classifier
detector2 = cv2.CascadeClassifier(detector2Path)
# dlibs CNN based classifier
# currently too slow even with rescaling
detector3 = dlib.cnn_face_detection_model_v1(detector3Path)
# dlibs facial landmark detector (68 points)
predictor = dlib.shape_predictor(shapePredPath)	

# basePath = "D:\\blink-detection\\data\\"
# blink = np.loadtxt(basePath + "onsim.csv",delimiter=',', skiprows=1)
# blink_labels = np.loadtxt(basePath + 'onsim_labels.csv',delimiter=',')

# #scaler = sk.StandardScalar()
# #scalar.fit(np.atleast_2d(bl#ink.T))
# # pd.Series(scalar.transform(np.atleast_2d(blink[1000:1200]).T[:,0]).plot())
# # pd.Series(blink_lables[1000:12000]).plot()

# ears = scipy.stats.zscore(blink[:,2])
# d_ears = np.gradient(ears)
# #ears = normalize(ears[:,np.newaxis],axis=0).ravel()
# #blink_labels = normalize(blink_labels[:,np.newaxis],axis=0).ravel()

# # pd.Series(ears).plot()
# # pd.Series(d_ears).plot()
# # pd.Series(blink_labels).plot()
# # plt.show()

# fname = "D:\\blink-detection\\vids\\rv3.MP4"
# show_landmarks(fname)

out = show_landmarks_webcam()
print(out)
#np.savetxt(csvPath + "out_landmarks.csv", out, delimiter=",")

