# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
	
# initial first transformation 
def horizontal_flip(src):
	return cv2.flip(src,-1)
	
# returns: 
# rect = dlib.rectangle or None if no faces found
# detectType to say which detector the rectangle represents
#
# if detector does not find a face than detector2 will try
def detectFaces(gray, detector,detector2):
	detectType = 1
	
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	# note rect is a dlib rectangle
	if(len(rects) == 0):
		# if no face found, try second detector
		faces = detector2.detectMultiScale(gray,1.3,5)
		if(len(faces) == 0):
			return None
		face = faces[0]
		(x,y,w,h) = face
		rect = dlib.rectangle(left=(x+w).item(), top=(y+h).item(), right=x.item(), bottom=y.item())	
		detectType = 2
	else:
		# use first face found
		rect = rects[0]
		
	return rect, detectType

def classifyVid(filename, detector, detector2, predictor, SHOW_FRAME = True):

	# output string for CSV file
	csv_out =  "Timestamp,Blink,FL_37,FL_38,FL_39,FL_40,FL_41,FL_42,"
	csv_out += "FL_43,FL_44,FL_45,FL_46,FL_47,FL_48\n"

	print(csv_out)
	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# define two constants, one for the eye aspect ratio to indicate
	# blink and then a second constant for the number of consecutive
	# frames the eye must be below the threshold
	EYE_AR_THRESH = 0.18
	EYE_AR_CONSEC_FRAMES = 2

	# initialize the frame counters and the total number of blinks
	COUNTER = 0
	TOTAL = 0
	
	# start the video stream thread
	print("[INFO] starting video stream thread...")
	#vs = FileVideoStream(vidPath,horizontal_flip).start()
	#vs = FileVideoStream(vidPath).start()
	vs = cv2.VideoCapture(vidPath + filename)
	
	fileStream = True
	# vs = VideoStream(src=0).start()
	# vs = VideoStream(usePiCamera=True).start()
	# fileStream = False
	time.sleep(1.0)
	FRAME_NUM = 0
	print("Applying classifer to " + vidPath + filename)
	

	print("Video properties:")
	FPS = vs.get(cv2.CAP_PROP_FPS)
	FC = vs.get(cv2.CAP_PROP_FRAME_COUNT)
	F = vs.get(cv2.CAP_PROP_FORMAT)
	print("\tFPS = " + str(FPS))
	print("\tFrame count = " + str(FC))
	print("\tFormat = " + str(F))
	# print("timestamp = 0")
	
	NOT_GRABBED = 0
	
	# loop over frames from the video stream
	# stop when we haven't grabbed 20 frames
	while NOT_GRABBED < 20:
		
		# Reset blink flag
		BLINKED = False
	
		# Try to grab frame
		(grabbed,frame) = vs.read()
		
		if(grabbed):
			print("Grabbed frame: " + str(FRAME_NUM) + "/" + str(FC))
			FRAME_NUM += 1
			NOT_GRABBED = 0
		else:
			print("Did not grab frame")
			NOT_GRABBED += 1
			continue
	
		# output timestamp and frame
		timestamp = 1000.0 * float(FRAME_NUM)/FPS
		#print("timestamp = " + str(timestamp), end=", ")
		#print("frame = " + str(FRAME_NUM))
		
		# A smaller resolution means faster results. 
		# A larger resolutions means better, slower results.	
		# frame = imutils.resize(frame)
		
		# Apply transformations (one of them being grayscale)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rect, detectType = detectFaces(gray,detector,detector2)
		
		if(detectType == 2):
			print("\tWriting detectType!")
			cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(),rect.bottom()),cv2.COLOR_BGR2HSV,10)
			cv2.imwrite(basePath + 'openCVOnly.png',frame)
		
		if(rect is not None):
			shape = predictor(gray,rect)
			shape = face_utils.shape_to_np(shape)
			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)

			cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(),rect.bottom()),cv2.COLOR_BGR2HSV,10)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
			
			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1
			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
					BLINKED = True

				# reset the eye frame counter
				COUNTER = 0
				
		if(SHOW_FRAME):
			display_frame = cv2.resize(frame,(480,320))
			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			# Reset counter if we did not find any faces
			if(rect is None):
				COUNTER = 0
				cv2.putText(display_frame, "EAR: N\\A", (300, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(display_frame, "NO FACE DETECTED", (300, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				cv2.putText(display_frame, "EAR: {:.2f}".format(ear), (300, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
				cv2.putText(display_frame, "Blinks: {}".format(TOTAL), (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			cv2.putText(display_frame, "Frame: {}".format(FRAME_NUM), (10, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
			# show the frame
			cv2.imshow("Frame", display_frame)
		
		# write to string
		# csv_out += str(int(timestamp)) + "," + str(int(BLINKED))
		# for coord in shape[lStart:lEnd]:
			# csv_out += "," + str(coord)
		# for coord in shape[rStart:rEnd]:
			# csv_out += "," + str(coord)
		# csv_out += "\n"
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
				

	# do a bit of cleanup
	cv2.destroyAllWindows()
	
	return [csv_out,TOTAL]
	
# Classifies a single picture
def classifyPic(picPath):

	# Load image in greyscale
	frame = cv2.imread(basePath + picPath,1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rect = detector(gray,0)[0]

	shape = predictor(gray,rect)
	shape = face_utils.shape_to_np(shape)
	# extract the left and right eye coordinates, then use the
	# coordinates to compute the eye aspect ratio for both eyes
	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]
	leftEAR = eye_aspect_ratio(leftEye)
	rightEAR = eye_aspect_ratio(rightEye)

	# average the eye aspect ratio together for both eyes
	ear = (leftEAR + rightEAR) / 2.0

	# compute the convex hull for the left and right eye, then
	# visualize each of the eyes
	leftEyeHull = cv2.convexHull(leftEye)
	rightEyeHull = cv2.convexHull(rightEye)

	cv2.rectangle(frame,(rect.left(),rect.top()),(rect.right(),rect.bottom()),cv2.COLOR_BGR2HSV,10)
	cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
	
	cv2.imshow("Frame", frame)
# ---------------------------------------------------------------------------
	
basePath = "D:\\blink-detection\\"
detector2Path = basePath + "haarcascade_frontalface_default.xml"
shapePredPath = basePath + "shape_predictor_68_face_landmarks.dat"
vidPath = basePath + "vids\\"
csvPath = basePath + "logs\\"
picPath = basePath + "pics\\"


WRITE_TO_CSV = False
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
detector2 = predictor2 = cv2.CascadeClassifier(detector2Path)
predictor = dlib.shape_predictor(shapePredPath)	

# Classify test picPath
# classifyPic(picPath + "test.png")

# Classify each video
for filename in os.listdir(vidPath):
	print(filename)
	if(filename.endswith(".MP4") or filename.endswith(".mp4")):
		vidName, ext = os.path.splitext(os.path.basename(filename))
		
		print("Classifying " + vidPath + filename)
		csv_out, total = classifyVid(filename,detector,detector2,predictor,SHOW_FRAME = True)
		print("total = " + str(total))
		
		if(WRITE_TO_CSV):
			print("Writing " + str(vidName) + ".csv")
			#write to csv file
			with open(csvPath + vidName + ".csv", "w") as text_file:
				text_file.write(csv_out)
				
		print("-----")
		
print("Done!")
