import os
import cv2
import time
import numpy as np

# flip frame
def horizontal_flip(src):
	return cv2.flip(src,-1)

# starts and stops the video to find the sychronization point.
def find_sync(video,bindings=0):

	buttons = {}
	if(bindings == 0):
		buttons["pause"] = ord('o')
		buttons["reverse"] = ord('k')
		buttons["quit"] = ord('q')
	if(bindings == 1):
		buttons["pause"] = ord('p')
		buttons["reverse"] = ord('l')
		buttons["quit"] = ord('w')

	vs = cv2.VideoCapture(vidPath + video)

	num_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
	FPS = vs.get(cv2.CAP_PROP_FPS)

	video_label = np.zeros(int(num_frames))
	current_frame = 0
	video_frame = []

	# FILL BUFFER
	while True:
		flag, frame = vs.read()
		print(current_frame)
		if flag:
			frame = cv2.resize(frame,(720,480))
			video_frame.append(frame)
			cv2.putText(frame, "Frame: {}".format(current_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(40)
		
			# pause
			if key == buttons["pause"]:
				print("paused!")
				frame_counter = len(video_frame) - 1
				frame_pause = video_frame[int(frame_counter)]
				while True:
					key2 = cv2.waitKey(0)
					if key2 == buttons["reverse"]:
						if frame_counter > 0:
							frame_counter -= 1
						frame_pause = video_frame[int(frame_counter)]
						cv2.putText(frame_pause, "Paused".format(frame_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						cv2.imshow("Frame", frame_pause)
					elif key2 == buttons["pause"]:
						break
					else:
						if frame_counter < len(video_frame) - 1:
							frame_counter += 1
						frame_pause = video_frame[int(frame_counter)]
						cv2.putText(frame_pause, "Paused".format(frame_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						cv2.imshow("Frame", frame_pause)
			elif key == buttons["quit"] :
				current_frame += 1
				break
				
			current_frame += 1
		else:
			vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
			print("frame is not ready")
			# It is better to wait for a while for the next frame to be ready
			time.sleep(1.0)
		
		if current_frame >= 1*int(FPS):
			# If the number of captured frames is equal to the total number of frames,
			# we stop
			break

	while True:
		flag, frame = vs.read()
		if flag:
			frame = cv2.resize(frame,(720,480))
			video_frame.append(frame)
			video_frame.pop(0)
			cv2.putText(frame, "Frame: {}".format(current_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(40)

			# pause
			if key == buttons["pause"]:
				frame_counter = len(video_frame) - 1
				frame_pause = video_frame[int(frame_counter)]
				while True:
					key2 = cv2.waitKey(0)
					if key2 == buttons["reverse"]:
						if frame_counter > 0:
							frame_counter -= 1
						frame_pause = video_frame[int(frame_counter)]
						cv2.putText(frame_pause, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						cv2.imshow("Frame", frame_pause)
					elif key2 == buttons["pause"]:
						break
					else:
						# any other goes forward a frame
						if frame_counter < len(video_frame) - 1:
							frame_counter += 1
						frame_pause = video_frame[int(frame_counter)]
						cv2.putText(frame_pause, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						cv2.imshow("Frame", frame_pause)
			elif key == buttons["quit"] :
				break
			
			current_frame += 1
		else:
			vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame-1)
			print("frame is not ready")
			# It is better to wait for a while for the next frame to be ready
			time.sleep(1.0)
		
		if current_frame == num_frames:
			# If the number of captured frames is equal to the total number of frames,
			# we stop
			break
		
	cv2.destroyAllWindows()
	vs.release()
	
	
# uses a1's rows, if a1 did not find a face (i.e empty) uses a2 row
def sync(csv1,csv2):
	
	
basePath = os.path.dirname(os.path.realpath(__file__)) + "\\"

vidPath = basePath + "vids\\"
csvPath = basePath + "logs\\"
picPath = basePath + "pics\\"
preTrainedPath = basePath + "pre_trained_models\\"

# get files
files = []
for filename in os.listdir(vidPath):
	print("loaded: " + filename)
	if(filename.endswith(".MP4") or filename.endswith(".mp4") or filename.endswith(".avi")):
		files.append(filename)

print(files)		
# sync videos
# find_sync(files[0])
