import os
import cv2
import time
import numpy as np

from numpy import nan

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
	
	
# arguments: two numpy arrays
# returns a3 such that if a1[row] has nan then a1[row] = a2[row] if a2[row] is not nan
def sync_arrays(a1,a2,SYNC_FRAME):
	
	t1 = a1[:,0]
	t2 = a2[:,0]
	assert SYNC_FRAME in t1 or SYNC_FRAME in t2

	# remove everything before sync frame
	index1 = list(t1).index(SYNC_FRAME)
	index2 = list(t2).index(SYNC_FRAME)
	a1 = a1[index1:len(a1)]
	a2 = a2[index2:len(a2)]
	
	# see returns
	x = np.isnan(a1)
	x = np.where(x)
	x = np.stack(x,axis=1)
	x = x[:,0]
	a1[x] = a2[x]
	
	return a1
	

	
# imports csv into numpy array and returns numpy array
def sync_csv(csv1,csv2,SYNC_FRAME):
	return sync_arrays(np.loadtxt(csv1),np.loadtxt(csv2))
	
basePath = os.path.dirname(os.path.realpath(__file__)) + "\\"

vidPath = basePath + "vids\\"
csvPath = basePath + "logs\\"
picPath = basePath + "pics\\"
preTrainedPath = basePath + "pre_trained_models\\"

# # get files
# files = []
# for filename in os.listdir(vidPath):
	# print("loaded: " + filename)
	# if(filename.endswith(".MP4") or filename.endswith(".mp4") or filename.endswith(".avi")):
		# files.append(filename)


# lets do some testing
a1 = np.asarray([[1,1.0],
[1,1.0],
[1,1.0],
[4,nan],
[5,1.0],
[6,1.0],
[7,1.0],
[8,nan]])
				 
a2 = np.asarray([[1,1.0],
[1,5.0],
[4,5.0],
[4,5.0],
[5,5.0],
[6,5.0],
[7,5.0],
[8,5.0]])

print(a1)
out = sync_arrays(a1,a2,1)
print(out)
		
# sync videos
# find_sync(files[0])
