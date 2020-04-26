# %% IMPORTS
import numpy as np
import cv2
import glob
from sklearn.neighbors import NearestNeighbors

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
h, w = cv2.imread(images_left[0]).shape[:2]

# vector representing the direction of the conveyor belt
conveyor_direction = np.array([[-667,225]])
conveyor_direction = conveyor_direction / np.linalg.norm(conveyor_direction)

# initialize region of interest
roi = [(0,0),(w,h)]
offset = 40

# for each frame of the video:
for i in range(n_images):
	# grab current frame and extract features
	frame_rgb_curr = cv2.imread(images_left[i])
	frame_gray_curr = cv2.cvtColor(frame_rgb_curr, cv2.COLOR_RGB2GRAY)
	features_curr = cv2.goodFeaturesToTrack(frame_gray_curr, maxCorners=1000, qualityLevel=0.01, minDistance=5)

	# if it's not the first frame:
	if (i>0):
		# find previous frame's features in the current frame
		features_flow, flow_status, flow_error = cv2.calcOpticalFlowPyrLK(frame_gray_prev, frame_gray_curr, features_prev, None)
		
		# first outlier removal: only keep points with positive status and small error
		mask1 = np.logical_and(flow_status==1,flow_error<3).squeeze()
		features_flow_good = features_flow[mask1]
		features_prev_good = features_prev[mask1]
		
		# create vector containing each feature's motion vector
		flow_vectors = features_flow_good-features_prev_good
		
		# second outlier removal: only keep points which moved, and whose direction is aligned with the conveyor belt's direction
		flow_vectors_magnitude = np.apply_along_axis(np.linalg.norm, 1, flow_vectors.squeeze())
		flow_vectors_projection = np.einsum("ij,ij->i", flow_vectors.squeeze(), np.repeat(conveyor_direction,flow_vectors.shape[0],axis=0))
		mask2 = np.logical_and(flow_vectors_magnitude>1, flow_vectors_projection>0.95*flow_vectors_magnitude)
		features_flow_good = features_flow_good[mask2]
		features_prev_good = features_prev_good[mask2]
		
		if (features_flow_good.shape[0] >=5):
			# third outlier removal: remove points whose average distance to the closest neighbours is too big
			nbrs = NearestNeighbors(n_neighbors=int(features_flow_good.shape[0]/2), algorithm='ball_tree').fit(features_flow_good.squeeze())
			distances, _ = nbrs.kneighbors(features_flow_good.squeeze())
			distances_avg = distances[:,1:].mean(axis=1)
			mask3 = distances_avg < 100
			features_flow_good = features_flow_good[mask3]
			features_prev_good = features_prev_good[mask3]
		
		# if at least 5 points passed all the tests
		if (features_flow_good.shape[0] >= 5):
			# update region of interest			
			roi = [(int(features_flow_good[:,:,0].min()-offset),int(features_flow_good[:,:,1].min()-offset)),(int(features_flow_good[:,:,0].max()+offset),int(features_flow_good[:,:,1].max()+offset))]
			
			# color points that passed all the tests
			for i in range(features_flow_good.shape[0]):			
				cv2.circle(frame_rgb_curr, (features_flow_good[i][0][0], features_flow_good[i][0][1]), 5, (0, 255, 0), -1)
				
		else:
			# (in the future, will use kalman filter to update the region of interest)
			roi = [(0,0),(w,h)]
		
		# color rectangle of region of interest
		cv2.rectangle(frame_rgb_curr, roi[0], roi[1], (0,0,255), 3)
		# display result
		cv2.imshow("Tracking", frame_rgb_curr)
	
	# save current image and features
	frame_gray_prev = frame_gray_curr
	features_prev = features_curr

	if (cv2.waitKey(1) & 0xFF == 27): break  # esc to quit
	
cv2.destroyAllWindows()