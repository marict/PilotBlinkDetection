# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
from operator import itemgetter
from numpy import nan

import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys,os
import importlib
import pdb

from data_funcs import *

shape = None

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

        # use largest face found
        faces = sorted(faces,reverse = True,key=lambda x: x[2] * x[3])
        face = faces[0]
        rect = openCv_rect_to_dlib(face)
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

# tool built by Steven to label the video data
# p to pause, b to label, k to go back, any other button goes forward
# video is file name
def label_video(video,starting_frame = 0,FLIP = False):
    vs = cv2.VideoCapture(vidPath + video)

    num_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    FPS = vs.get(cv2.CAP_PROP_FPS)

    video_label = np.zeros(int(num_frames))
    current_frame = 0
    video_frame = []

    print("num frames = " + str(num_frames))

    if(starting_frame > num_frames):
        return "Starting_frame > num_frames"

    # go to starting frame
    while current_frame != starting_frame:
        cv2.waitKey(40)
        flag,frame = vs.read()
        if flag:
            if(current_frame % int(num_frames/100) == 0):
                print("loading: " + str(current_frame))
            current_frame += 1
        else:
            print("did not grab frame")
            time.sleep(0.5)

    first = True
    while True:
        #print("current frame: " + str(current_frame))
        flag, frame = vs.read()
        if flag:
            frame = cv2.resize(frame,(1280,720))
            if(FLIP):
                frame = horizontal_flip(frame)
            video_frame.append(frame)
            if(len(video_frame) >= 8*int(FPS)):
                video_frame.pop(0)
            cv2.putText(frame, "Frame: {}".format(current_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(40)

            # pause
            if first or key == ord('p'):
                first = False
                frame_counter = len(video_frame) - 1
                frame_pause = video_frame[int(frame_counter)]
                while True:
                    key2 = cv2.waitKey(0)
                    # label
                    if key2 == ord('b'):
                        video_label[current_frame - (len(video_frame) - frame_counter)] = 1
                        vidName, ext = os.path.splitext(os.path.basename(file))
                        np.savetxt(csvPath + vidName + "__openclosed_labels.csv", video_label, delimiter=",")                        
                        cv2.putText(frame_pause, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame_pause, "Labeled: {}".format(current_frame - (len(video_frame) - frame_counter-1)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame_pause, "Saved data", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # go forward a frame
                        if frame_counter < len(video_frame) - 1:
                            frame_counter += 1
                        frame_pause = video_frame[int(frame_counter)]
                        cv2.imshow("Frame", frame_pause)
                    # go back one frame
                    elif key2 == ord('k'):
                        if frame_counter > 0:
                            frame_counter -= 1
                        frame_pause = video_frame[int(frame_counter)]
                        cv2.putText(frame_pause, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame_pause)
                    elif key2 == ord('p'):
                        break
                    else:
                        # any other goes forward a frame
                        if frame_counter < len(video_frame) - 1:
                            frame_counter += 1
                        frame_pause = video_frame[int(frame_counter)]
                        cv2.putText(frame_pause, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame_pause)
            elif key == ord('q'):
                break    
            current_frame += 1
        else:
            # vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame-1)
            print("frame is not ready")
            key = cv2.waitKey(40)
            if(key == ord('q')):
                break
            # It is better to wait for a while for the next frame to be ready
            continue

        if current_frame == num_frames or current_frame == num_frames - 1:
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            key2 = cv2.waitKey(0)
            if(key == ord('q')):
                break

    cv2.destroyAllWindows()
    vs.release()
    return video_label

# NOT FULLY IMPLEMENTED: DOES OPENCV EYE DETECTION 
def gen_images(filename, SHOW_FRAME = True, FLIP = False):
    print("Generating images for " + vidPath + filename)

    # How much to downsample face for detection
    FACE_DOWNSAMPLE_RATIO = 4

    # output matrix
    out = []

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    vs = cv2.VideoCapture(vidPath + filename)

    time.sleep(5.0)
    FRAME_NUM = 0
    print("Applying classifer to " + vidPath + filename)

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
            print("not grabbed = " + str(NOT_GRABBED))
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
            face = frame[rect_resize.top():rect_resize.bottom(),rect_resize.left():rect_resize.right()]
            eyes = detector4_eyes.detectMultiScale(face)

            # draw eyes
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            shape = predictor(gray,rect_resize)
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

            cv2.rectangle(frame,(rect_resize.left(),rect_resize.top()),(rect_resize.right(),rect_resize.bottom()),cv2.COLOR_BGR2HSV,10)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)


        if(SHOW_FRAME):
            display_frame = cv2.resize(frame,(1280,720))
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            # Reset counter if we did not find any faces
            if(rect is None):
                cv2.putText(display_frame, "EAR: N\\A", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "NO FACE DETECTED", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display_frame, "Frame: {}".format(FRAME_NUM), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # show the frame
            cv2.imshow("Frame", display_frame)

        # output different csv things if face was detected
        if(rect is not None):
            out.append([int(timestamp),ear])
        else:
            out.append([int(timestamp),nan])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    
# Generates a csv string of the ears of each frame (nothing if no face detected)
def gen_ears(filename, SHOW_FRAME = True, FLIP = False):
    print("Generating ears  for " + vidPath + filename)

    # How much to downsample face for detection
    FACE_DOWNSAMPLE_RATIO = 4

    # output matrix
    out = []

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    vs = cv2.VideoCapture(vidPath + filename)

    time.sleep(5.0)
    FRAME_NUM = 0
    print("Applying classifer to " + vidPath + filename)

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
            print("not grabbed = " + str(NOT_GRABBED))
            NOT_GRABBED += 1
            continue

        if(FLIP):
            frame = horizontal_flip(frame)

        # output timestamp and frame
        #timestamp = 1000.0 * float(FRAME_NUM)/FPS
        timestamp = FRAME_NUM

        # Apply transformations (one of them being grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rect, detectType  = detectFaces(gray, FACE_DOWNSAMPLE_RATIO)

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

            cv2.rectangle(frame,(rect_resize.left(),rect_resize.top()),(rect_resize.right(),rect_resize.bottom()),cv2.COLOR_BGR2HSV,10)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)


        if(SHOW_FRAME):
            display_frame = cv2.resize(frame,(1280,720))
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            # Reset counter if we did not find any faces
            if(rect is None):
                cv2.putText(display_frame, "EAR: N\\A", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "NO FACE DETECTED", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display_frame, "Frame: {}".format(FRAME_NUM), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # show the frame
            cv2.imshow("Frame", display_frame)

        # output different csv things if face was detected
        if(rect is not None):
            out.append([int(timestamp),ear])
        else:
            out.append([int(timestamp),nan])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return out
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()

    return np.asarray(out)

# Generates a csv string of the landmarks of each frame (nothing if no face detected)
def gen_landmarks(filename, SHOW_FRAME = True, FLIP = False):
    print("Generating landmarks  for " + vidPath + filename)

    # How much to downsample face for detection
    FACE_DOWNSAMPLE_RATIO = 4

    # output matrix
    out = []

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    vs = cv2.VideoCapture(vidPath + filename)

    time.sleep(5.0)
    FRAME_NUM = 0
    print("Applying classifer to " + vidPath + filename)

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
            print("not grabbed = " + str(NOT_GRABBED))
            NOT_GRABBED += 1
            continue

        if(FLIP):
            frame = horizontal_flip(frame)

        # output timestamp and frame
        #timestamp = 1000.0 * float(FRAME_NUM)/FPS
        timestamp = FRAME_NUM

        # Apply transformations (one of them being grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rect, detectType  = detectFaces(gray, FACE_DOWNSAMPLE_RATIO)

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
            
            cv2.rectangle(frame,(rect_resize.left(),rect_resize.top()),(rect_resize.right(),rect_resize.bottom()),cv2.COLOR_BGR2HSV,10)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
            cv2.circle(frame, totuple(shape[28]), 1, (255,0,0),2)
            
            # adjust shapes so that middle (28) is centered
            centered_shape = shape - shape[28] + [frame.shape[1]/2,frame.shape[0]/2]
            centered_shape = centered_shape.astype(np.uint64)
            cv2.polylines(frame, [centered_shape], 1, (0, 0, 255), 2)
        
        # output different csv things if face was detected
        if(rect is not None):
            out.append(centered_shape.flatten())
        else:
            out.append(np.zeros(136))

        if(SHOW_FRAME):
            display_frame = cv2.resize(frame,(1280,720))
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            # Reset counter if we did not find any faces
            if(rect is None):
                cv2.putText(display_frame, "EAR: N\\A", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "NO FACE DETECTED", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display_frame, "Frame: {}".format(FRAME_NUM), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # show the frame
            cv2.imshow("Frame", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return out
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    return np.asarray(out)
    
# ---------------------------------------------------------------------------

WRITE_TO_CSV = True

print("[INFO] loading facial classifiers...")
# dlibs HOG based classifier
detector = dlib.get_frontal_face_detector()
# openCVs HAAR cascade classifier
detector2 = cv2.CascadeClassifier(detector2Path)
# dlibs CNN based classifier
# currently too slow even with rescaling, requires version 1.9 of dlib
# detector3 = dlib.cnn_face_detection_model_v1(detector3Path)

# openCV's eye detector
detector4_eyes = cv2.CascadeClassifier(detector4Path)

# dlibs facial landmark detector (68 points)
predictor = dlib.shape_predictor(shapePredPath)

# get files
files = []
for filename in os.listdir(vidPath):
    print(filename)
    if(filename.endswith(".MP4") or filename.endswith(".mp4")):
        files.append(filename)
        
# gen landmarks
file = files[0]
out = gen_landmarks(file)
out = np.asarray(out)
print(out.shape)
vidName, ext = os.path.splitext(os.path.basename(file))
np.savetxt(csvPath + vidName + "_landmarks.csv", out, delimiter=",")
		
# # gen ears
# for file in files:
	# vidName, ext = os.path.splitext(os.path.basename(file))
	# out = gen_ears(file,SHOW_FRAME = True)
	# if(WRITE_TO_CSV):
		# print("Writing " + str(vidName) + "_ears.csv")
		# #np.savetxt(csvPath + vidName + "_ears.csv", out, delimiter=",")
			
	# print("-----")


# # Label video
# file = files[0]
# out = label_video(file)
# print(out.shape)
# vidName, ext = os.path.splitext(os.path.basename(file))
# np.savetxt(csvPath + vidName + "_openclosed_labels_13000.csv", out, delimiter=",")
	

print("Done!")
