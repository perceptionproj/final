# %% IMPORTS
import numpy as np
import cv2
import glob

# %% LOAD IMAGES
dir_dataset = '../datasets/conveyor_without_occlusions'
images_left = glob.glob(dir_dataset + '/left/*.png')
images_right = glob.glob(dir_dataset + '/right/*.png')

assert images_right, images_left
assert (len(images_right) == len(images_left))
n_images = len(images_right)
images_right.sort()
images_left.sort()

# %% LOAD CALIBRATION MATRICES AND CAMERA PARAMETERS
dir_params = "../calibration/calibration_matrix/"
mtx_l = np.load(dir_params + "camera_matrix_l.npy")
dist_l = np.load(dir_params + "distortion_coeff_l.npy")
mtx_l_new = np.load(dir_params + "camera_matrix_l_new.npy")
mtx_r = np.load(dir_params + "camera_matrix_r.npy")
dist_r = np.load(dir_params + "distortion_coeff_r.npy")
mtx_r_new = np.load(dir_params + "camera_matrix_r_new.npy")
mtx_R = np.load(dir_params + "rotation_l_r.npy")
mtx_T = np.load(dir_params + "translation_l_r.npy")
mtx_E = np.load(dir_params + "essential_matrix.npy")
mtx_F = np.load(dir_params + "fundamental_matrix.npy")
mtx_P_l = np.load(dir_params + "projection_matrix_l.npy")
mtx_P_r = np.load(dir_params + "projection_matrix_r.npy")

#%% MOTION DETECTION

for i in range(n_images):
	
	# grab current frame and extract features
	frame_rgb_curr = cv2.imread(images_left[i])
	frame_gray_curr = cv2.cvtColor(frame_rgb_curr, cv2.COLOR_RGB2GRAY)
	features_curr = cv2.goodFeaturesToTrack(frame_gray_curr, maxCorners=1000, qualityLevel=0.01, minDistance=5)

	if (i>0):
		# calculate optical flow
		features_flow, flow_status, flow_error = cv2.calcOpticalFlowPyrLK(frame_gray_prev, frame_gray_curr, features_prev, None)
		# display moving points in green and static points in red
		for i in range(len(features_flow)):
			flow_vector = np.array([features_flow[i][0][0]-features_prev[i][0][0],features_flow[i][0][1]-features_prev[i][0][1]])							  
			conveyor_direction = np.array([-0.9578,0.2873])
			if (np.linalg.norm(flow_vector)>1 and np.dot(flow_vector,conveyor_direction) > (0.99*np.linalg.norm(flow_vector))):
				cv2.circle(frame_rgb_curr, (features_flow[i][0][0], features_flow[i][0][1]), 5, (0, 255, 0), -1)
			else:
				cv2.circle(frame_rgb_curr, (features_flow[i][0][0], features_flow[i][0][1]), 2, (0, 0, 255), -1)
		
		cv2.imshow("Tracking", frame_rgb_curr)
	
	# save previous image and features
	frame_gray_prev = frame_gray_curr
	features_prev = features_curr

	if (cv2.waitKey(1) & 0xFF == 27): break  # esc to quit
	
cv2.destroyAllWindows()