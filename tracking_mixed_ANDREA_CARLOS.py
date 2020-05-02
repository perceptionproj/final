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
dir_calib = "../calibration/calibration_matrix/"
mtx_P_l = np.load(dir_calib + "projection_matrix_l.npy")
mtx_P_r = np.load(dir_calib + "projection_matrix_r.npy")
rect_map_l_x = np.load(dir_calib + "map_l_x.npy")
rect_map_l_y = np.load(dir_calib + "map_l_y.npy")
rect_map_r_x = np.load(dir_calib + "map_r_x.npy")
rect_map_r_y = np.load(dir_calib + "map_r_y.npy")

# %% SETTINGS
h, w = cv2.imread(images_left[0]).shape[:2] # size of the images (pixels)

roi_start = np.array([1030,240,1270,440]) # initialized region of interest where the object is expected to appear (pixels)
roi_padding = 40 # expansion of the region of interest to allow for a dynamical region of interest (pixels)

conveyor_x0 = 470 # x beginning of the conveyor (pixels)
conveyor_x1 = 1170 # x end of the conveyor (pixels)

conveyor_direction = np.array([[-725,236]]) # vector representing the direction of the conveyor belt
conveyor_direction = conveyor_direction / np.linalg.norm(conveyor_direction) # normalized vector

of_max_objs = 1000 # maximun number of features to find
of_quality = 0.03 # quality of the features (lower -> more features)
of_min_dist = 3 # minumum distance between the features (pixels)
of_err_threshold = 3 # error threshold of the features (discard features with higher value)

of_magnitude = 1 # minimun speed for a feature point to be considered (pixels/frame)
of_projection = 0.99 # discard features whose velocity vector, projected on the direction of the conveyor, retains less than 0.95 of their magnitude (they are not moving on the conveyor)

of_ngb_dist_max = 300 # maximun medium distance to the feature's nearest neighbours (pixels)

of_samples_mean = 5 # number of consecutive frames to consider for the mean of the roi position (higher = smoother)

min_N = 5 # minumin number of points that have to pass all the outlier tests in order for the measurement to be considered

# %% TRACKING

# initialize frame_prev and features_prev
features_prev = np.empty((0,1,2),dtype='float32')
frame_gray_prev = np.zeros((h,w),dtype='uint8')

# initialize roi
roi = roi_start
roi_hist = np.asarray(roi_start).reshape(1,4)

# initialize blackground subtractor
fgbg = cv2.createBackgroundSubtractorKNN(history=600,dist2Threshold=800, detectShadows=False)
# define kernels
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# initialize object count and status
obj_count = 0
obj_present = False # (is there an object on the scene?)
obj_found = False # (was it possible to localize the object on the scene?)


# for each frame of the video:
for i in range(n_images):
	# grab current frame and extract features
	frame_rgb_curr = cv2.imread(images_left[i])
	frame_rgb_curr_right = cv2.imread(images_right[i])
	frame_gray_curr = cv2.cvtColor(frame_rgb_curr, cv2.COLOR_RGB2GRAY)
	frame_gray_curr_right = cv2.cvtColor(frame_rgb_curr_right, cv2.COLOR_RGB2GRAY)
	
	# undistort and rectify left and right image
	frame_rgb_curr = cv2.remap(frame_rgb_curr, rect_map_l_x, rect_map_l_y, cv2.INTER_LINEAR)
	frame_rgb_curr_right = cv2.remap(frame_rgb_curr_right, rect_map_r_x, rect_map_r_y, cv2.INTER_LINEAR)
	frame_gray_curr = cv2.remap(frame_gray_curr, rect_map_l_x, rect_map_l_y, cv2.INTER_LINEAR)
	frame_gray_curr_right = cv2.remap(frame_gray_curr_right, rect_map_r_x, rect_map_r_y, cv2.INTER_LINEAR)
	
	# apply the blackground subtration with the current frame
	fgmask = fgbg.apply(frame_rgb_curr)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)

	# create mask for goodFeaturesToTrack (only look for features that are inside the region of interest)
	mask_roi = np.zeros((h,w),dtype='uint8')
	mask_roi[roi[1]:roi[3],roi[0]:roi[2]] = 255
	
	# combine region of interest and blackground subtractor
	mask_object = cv2.bitwise_and(mask_roi, fgmask)

	# find good features to track
	features_curr = cv2.goodFeaturesToTrack(frame_gray_curr, maxCorners=of_max_objs, qualityLevel=of_quality, minDistance=of_min_dist, mask=mask_object)

	# if it's not the first frame:
	if (i>0):
		features_flow, flow_status, flow_error = cv2.calcOpticalFlowPyrLK(frame_gray_prev, frame_gray_curr, features_prev, None)
		if (features_flow is None):
			features_flow = np.empty((0,1,2),dtype='float32')
		# first outlier removal: only keep points with positive status and small error
		if (features_flow.shape[0] > min_N):
			mask1 = np.logical_and(flow_status==1,flow_error<of_err_threshold)[:,0]
			features_flow = features_flow[mask1]
			features_prev = features_prev[mask1]
		
		# second outlier removal: only keep points that moved, and whose velocity vector is aligned with the conveyor belt's direction
		if (features_flow.shape[0] > min_N):
			flow_vectors = features_flow-features_prev # vector containing each feature's motion vector
			flow_vectors_magnitude = np.apply_along_axis(np.linalg.norm, 1, flow_vectors[:,0,:])
			flow_vectors_projection = np.einsum("ij,ij->i", flow_vectors[:,0,:], np.repeat(conveyor_direction,flow_vectors.shape[0],axis=0))
			mask2 = np.logical_and(flow_vectors_magnitude>of_magnitude, flow_vectors_projection>of_projection*flow_vectors_magnitude)
			features_flow = features_flow[mask2]
		
		# third outlier removal: remove points whose average distance to the closest neighbours is too big
		if (features_flow.shape[0] >= min_N):
			nbrs = NearestNeighbors(n_neighbors=int(features_flow.shape[0]/2), algorithm='ball_tree').fit(features_flow[:,0,:])
			distances, _ = nbrs.kneighbors(features_flow[:,0,:])
			distances_avg = distances[:,1:].mean(axis=1)
			mask3 = distances_avg < of_ngb_dist_max
			features_flow = features_flow[mask3]
	
		# if at least 5 points passed all the tests (VALID MEASUREMENT):
		if (features_flow.shape[0] >= min_N):
			
			# calculate center of object
			features_flow_center = (int(features_flow[:,:,0].mean()),int(features_flow[:,:,1].mean()))

			# make sure that the x an y coordinates of the center of the object is in the region of the conveyor:
			if (features_flow_center[0]<=conveyor_x1 and features_flow_center[0]>=conveyor_x0):
				
				obj_found = True
				
				if (obj_present is False):
					obj_present = True
					obj_count += 1
				
				# update region of interest
				roi = np.array([int(features_flow[:,:,0].min()-2*roi_padding),int(features_flow[:,:,1].min()-roi_padding),int(features_flow[:,:,0].max()+roi_padding),int(features_flow[:,:,1].max()+roi_padding)])
				roi_hist = np.vstack((roi_hist,roi))[-of_samples_mean:,:]
				# draw center of the object
				roi_hist_center = (int((roi_hist.mean(axis=0)[2]+roi_hist.mean(axis=0)[0])/2),int((roi_hist.mean(axis=0)[3]+roi_hist.mean(axis=0)[1])/2))
				cv2.circle(frame_rgb_curr, roi_hist_center, 6, (0, 255, 0), -1)
				cv2.circle(frame_rgb_curr, features_flow_center, 6, (255,0,0), -1)
				# draw points that passed all the tests
				for i in range(features_flow.shape[0]):			
					cv2.circle(frame_rgb_curr, (features_flow[i][0][0], features_flow[i][0][1]), 2, (0, 255, 0), -1)

			# if the object has reached the end of the conveyor:
			elif (features_flow_center[0]<conveyor_x0):
				obj_present = False
				obj_found = False
				# initialize roi (wait for new object to appear)
				roi = roi_start
				roi_hist = np.asarray(roi_start).reshape(1,4)
				# reset features to track (find new ones)
				features_curr = np.empty((0,1,2),dtype='float32')

				
			# if the object is too much on the right:
			elif (features_flow_center[0]>conveyor_x0):
				# initialize roi (wait for object to appear)
				roi = roi_start
				roi_hist = np.asarray(roi_start).reshape(1,4)

		# if LESS THAN 5 points passed all the tests (INVALID MEASUREMENT):
		else:
			obj_found = False
			
			if (obj_present is True):
				# if the object is present and has not yet reached the end of the conveyor:
				if ((roi[2]+roi[0])/2 >= conveyor_x0):
					obj_present = True
					# update the pose with a (!FAKE!) Kalman filter (just shift the roi a little bit)
					roi = np.array([roi[0]-4,roi[1]+2,roi[2]-4,roi[3]+2])
					roi_hist = np.vstack((roi_hist,roi))[-of_samples_mean:,:]
					
	
				# if the object has reached the end of the conveyor:
				else:
					obj_present = False
					# initialize roi (wait for new object to appear)
					roi = roi_start
					roi_hist = np.asarray(roi_start).reshape(1,4)
					# reset features to track (find new ones)
					features_curr = np.empty((0,1,2),dtype='float32')
			
		# %% VISUALIZE RESULT
		# set some colors and text depending on the status of the object
		if obj_present is True:
			if obj_found is True:
				vis_text = "Tracking object " + str(obj_count)
				vis_box_color = (0,255,0)
			else:
				vis_text = "Lost object " + str(obj_count) + ". Predicting"
				vis_box_color = (0,255,255)
		else:
			vis_text = "Waiting for object " + str(obj_count+1)
			vis_box_color = (0,0,255)

		# write legend below roi
		cv2.putText(frame_rgb_curr, "Blue Point = features center", (roi_hist.mean(axis=0)[0].astype(int),roi_hist.mean(axis=0)[3].astype(int)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA )
		cv2.putText(frame_rgb_curr, "Green Point = roi center", (roi_hist.mean(axis=0)[0].astype(int),roi_hist.mean(axis=0)[3].astype(int)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA )
		# write object status above roi
		cv2.putText(frame_rgb_curr, vis_text, (roi_hist.mean(axis=0)[0].astype(int),roi_hist.mean(axis=0)[1].astype(int)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, vis_box_color, 1, cv2.LINE_AA)
		# draw rectangle of region of interest
		cv2.rectangle(frame_rgb_curr, tuple(roi_hist.mean(axis=0)[:2].astype(int)), tuple(roi_hist.mean(axis=0)[-2:].astype(int)), vis_box_color, 2)
		# display final result
		cv2.imshow("Tracking", frame_rgb_curr)
	
	if (cv2.waitKey(1) & 0xFF == 27): break  # ESC to quit
	
	# %% NEXT FRAME, PLEASE
	# save current image and features for next use
	frame_gray_prev = frame_gray_curr
	if (features_curr is not None):
		features_prev = features_curr
	else:
		features_prev = np.empty((0,1,2),dtype='float32')

cv2.destroyAllWindows()