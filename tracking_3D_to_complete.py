#%% IMPORT
import numpy as np
import cv2
import glob
import imutils
import time
import SIFT1_Classify as myClassifier

# %% LOAD CALIBRATION MATRICES AND CAMERA PARAMETERS
dir_calib = "../calibration/calibration_matrix/"
mtx_P_l = np.load(dir_calib + "projection_matrix_l.npy")
mtx_P_r = np.load(dir_calib + "projection_matrix_r.npy")
rect_map_l_x = np.load(dir_calib + "map_l_x.npy")
rect_map_l_y = np.load(dir_calib + "map_l_y.npy")
rect_map_r_x = np.load(dir_calib + "map_r_x.npy")
rect_map_r_y = np.load(dir_calib + "map_r_y.npy")

# %% LOAD IMAGES
dataset = '../datasets/conveyor_with_occlusions'
images_left = glob.glob(dataset + '/left/*.png')
images_right = glob.glob(dataset + '/right/*.png')

assert images_right, images_left
assert (len(images_right) == len(images_left))
n_images = len(images_right)
images_right.sort()
images_left.sort()

# %% FUNCTIONS DECLARATION
def getBlueMask(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	low_blue = np.array([105,40,95])
	up_blue = np.array([120,255,255])
	mask = cv2.inRange(hsv, low_blue, up_blue)
	kernel = np.ones((5,5),np.uint8)
	mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
	mask_morph = cv2.bitwise_not(mask_morph)	
	return mask_morph

def getRectangleCenter(p1,p2):
	x = int(p1[0] + (p2[0] - p1[0])/2)
	y = int(p1[1] + (p2[1] - p1[1])/2)
	return (x,y)


def updateAreaslist(area, areas, n_storedAreas):
	areas.insert(0, area)
	try:
		areas.pop(n_storedAreas)
	except:
		return

def AreaGettingSmaller(areas, n_storedAreas):
	if len(areas) != n_storedAreas:
		return False
	last_arrea = areas[0]
	mean_arrea = np.mean(areas[1:], axis=0)
	d_area = last_arrea - mean_arrea
	if -0.10 < d_area/mean_arrea < -0.03:
		# print("True:", d_area/mean_arrea)
		return True
	else:
		# print("False:", d_area/mean_arrea)
		return False

def get2DVel(coords):
	arr = np.asarray(coords)
	mean_x, mean_y = np.mean(arr, axis=0)
	vx = coords[0][0] - mean_x
	vy = coords[0][1] - mean_y
	return vx, vy
############################################

def drawTrackingInformation(mp_3d):
	x = round(mp_3d[0][0][0], 2)
	y = round(mp_3d[0][0][1], 2)
	z = round(mp_3d[0][0][2], 2)

	vis_tracking_y = 350
	cv2.putText(frame, "TRACKING:", (10,vis_tracking_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,150,0), 1, cv2.LINE_AA)
	cv2.putText(frame, "2D Camera Coordinates: ", (20,vis_tracking_y+35), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(frame, "x: "+str(center_rectangle[0]), (30,vis_tracking_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(frame, "y: "+str(center_rectangle[1]), (30,vis_tracking_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(frame, "3D World Trangulation: ", (20,vis_tracking_y+115), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(frame, "x: "+str(x), (30,vis_tracking_y+140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(frame, "y: "+str(y), (30,vis_tracking_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(frame, "z: "+str(z), (30,vis_tracking_y+180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

def	matchPoint(mp_left_frame, template_h, template_w, roi_h, roi_left_off, roi_right_off, frame, frame_right, gray, gray_right):
	# Region of interest to match the template within (Right frame coords)
	roi_triang = [(mp_left_frame[0]+int(roi_left_off), mp_left_frame[1]-int(roi_h/2)), 
		(mp_left_frame[0]+int(roi_right_off), mp_left_frame[1]+int(roi_h/2))]
	cv2.rectangle(frame_right, roi_triang[0], roi_triang[1], (255,0,0), 1)

	# Drawing tamplate's boundaries on the left frame
	cv2.rectangle(frame, (mp_left_frame[0]-int(template_w/2), mp_left_frame[1]-int(template_h/2)), 
		(mp_left_frame[0]+int(template_w/2), mp_left_frame[1]+int(template_h/2)), (255,0,0), 1)
	
	# Cropping the template
	template = gray[mp_left_frame[1]-int(template_h/2): mp_left_frame[1]+int(template_h/2), 
		mp_left_frame[0]-int(template_w/2): mp_left_frame[0]+int(template_w/2)]
	
	# Cropping the roi from the right frame
	cropped_roi_gray = gray_right[mp_left_frame[1]-int(roi_h/2):mp_left_frame[1]+int(roi_h/2), 
		mp_left_frame[0]+int(roi_left_off):mp_left_frame[0]+int(roi_right_off)]

	# Matching template
	res = cv2.matchTemplate(cropped_roi_gray, template, cv2.TM_CCORR_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	
	# Getting matched point position in roi coords
	top_left = max_loc
	mp_local = (top_left[0] + template.shape[1]/2, top_left[1] + template.shape[0]/2)

	# Matched point position in right frame coords
	mp_right_frame = (mp_local[0] + roi_triang[0][0], mp_local[1] + roi_triang[0][1])
	return mp_right_frame


# %% SETTINGS and VARIABLES DEFINITION
h, w = cv2.imread(images_left[0]).shape[:2] # size of the images (pixels)
min_area = 2500 # define minimum object area to avoid small outliers regions (pixels^2)

# define previous image
prev_img = cv2.imread(images_left[0])
prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

# initialize frame_prev and features_prev
feat_prev = np.empty((0,1,2),dtype='float32')
gray_prev = np.zeros((h,w),dtype='uint8')

# define region of the belt
belt_contour = np.array([[[387,476]],[[464,696]],[[1217,359]],[[1131,260]]])
belt_x0 = 400 # x start of the conveyor (pixels)
belt_x1 = 1240 # x end of the conveyor (pixels)

# mask to remove hands
mask_belt_x = np.zeros((h,w),dtype='uint8')
mask_belt_x[:,belt_x0:belt_x1] = 255

# define kernels
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# initialize background subtractor
fgbg = cv2.createBackgroundSubtractorKNN(history=600,dist2Threshold=800, detectShadows=False)

# initialize roi
point1_start = (1030,240)
point2_start = (1270,440)
point1 = point1_start
point2 = point2_start
center_rectangle = getRectangleCenter(point1, point2)
center_rectangle_prev = (0,0)

# initialize object count and status
obj_count = 0
object_on_conveyor = None
obj_present = False # (is there an object on the scene?)
obj_found = False # (was it possible to localize the object on the scene?)

# initialize object classification counter
obj_type_hist = {"cup":0,"book":0,"box":0}

# Triangulation constants
template_h = 10
template_w = 60
# roi within tamplate is being matched
roi_h = 10
roi_left_off = -230
roi_right_off = -30

# length of a list
n_2Dpositions = 10
# main list of 2D positions
coords1 = []
# backup list of 2D positions
coords2 = []
# list of the last areas
areas = []
# length of a list
n_storedAreas = 6
# occlusion params
gotOccluded = False
LastPosOccluded = False
JustGotOccluded = False
# Features ext constants
of_max_objs = 100
of_quality = 0.05
of_min_dist = 10
counter = 0
p0 = []
p1 = []

# Parameters for lucas kanade optical flow 
lk_params = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                              10, 0.03)) 


# %% MAIN

for i in range(1, n_images):

	# grab current frame 
	frame = cv2.imread(images_left[i])
	frame_right = cv2.imread(images_right[i])
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

	# undistort and rectify left and right image
	frame = cv2.remap(frame, rect_map_l_x, rect_map_l_y, cv2.INTER_LINEAR)
	frame_right = cv2.remap(frame_right, rect_map_r_x, rect_map_r_y, cv2.INTER_LINEAR)
	gray = cv2.remap(gray, rect_map_l_x, rect_map_l_y, cv2.INTER_LINEAR)
	gray_right = cv2.remap(gray_right, rect_map_r_x, rect_map_r_y, cv2.INTER_LINEAR)

	# apply the background subtractor to the current frame
	mask_fg = fgbg.apply(frame)

	# get blue mask that characterizes the conveyor belt
	mask_blue = getBlueMask(frame)
	
	# combine blue mask, background subtractor and hands mask
	mask_fg = cv2.bitwise_and(mask_fg, mask_belt_x)
	mask_fg = cv2.bitwise_and(mask_fg, mask_blue)
	mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=2)

	# find the contours
	cnts = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	obj_found = False
	obj_picture = np.zeros((1,1,3),dtype="uint8")
	obj_type = "N/A - Out of range"


	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,500)
	fontScale              = 1
	fontColor              = (255,255,255)
	lineType               = 2

	cv2.putText(gray,'n:'+str(i), 
		bottomLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType) 

	if len(cnts)>0:
		# get contour with the highest area
		c = max(cnts, key=cv2.contourArea)
		area = cv2.contourArea(c)
		
		# if the object is big enough (not noise)
		if area > min_area:          

			# compute the bounding box for the contour
			(x,y,width,height) = cv2.boundingRect(c)
			point1 = (x,y)
			point2 = (x+width, y+height)
			# compute center of the bounding box
			center_rectangle = getRectangleCenter(point1, point2)
			# compute movement vector of the center
			center_movement = np.array(center_rectangle)-np.array(center_rectangle_prev)
			
			# if the center didn't move too much (not a new object or noise)
			if (np.linalg.norm(center_movement)<200):
				# check that object is on the conveyor
				object_on_conveyor = cv2.pointPolygonTest(belt_contour, center_rectangle, measureDist = False)
				# if the object is on the conveyor:
				if object_on_conveyor==1.0:
					# if it's a new object:
					if not obj_present:
						# increase object count
						obj_count += 1
						# initialize classification counters
						obj_type_hist = {"cup":0,"book":0,"box":0}
					# update object status
					obj_present = True
					obj_found = True
					# CLASSIFICATION
					# extract object from current frame
					mask_roi = np.zeros((h,w),dtype='uint8')
					mask_roi[point1[1]:point2[1],point1[0]:point2[0]] = 255
					mask_obj = cv2.bitwise_and(mask_roi, mask_fg)
					obj_picture = cv2.bitwise_or(frame, frame, mask = mask_obj)
					obj_picture = obj_picture[point1[1]:point2[1],point1[0]:point2[0]]
					# classify object
					obj_type = myClassifier.detectAndClassify(obj_picture)
					obj_type_hist[obj_type] += 1
					
			# if the center moved too much (new object or noise)
			else:
				# reset roi
				point1 = point1_start
				point2 = point2_start
				center_rectangle = getRectangleCenter(point1, point2)

			updateAreaslist(area, areas, n_storedAreas)
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			if AreaGettingSmaller(areas, n_storedAreas) and LastPosOccluded == False and 1170 > cX > 900 and cY <900:
				gotOccluded = True
				JustGotOccluded = True
				counter += 1
				print(cX, cY)
				print ("occlusion", counter)
			else:
				JustGotOccluded = False
				pass

			if JustGotOccluded:
				p0 = cv2.goodFeaturesToTrack(gray, maxCorners=of_max_objs, qualityLevel=of_quality, 
					minDistance=of_min_dist, mask=mask_roi)
				corners = np.int0(p0)
				for i in corners:
					x,y = i.ravel()
					cv2.circle(frame,(x,y),3,255,-1)

			if gotOccluded and not JustGotOccluded:
				p1, flow_status, flow_error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, 
					p0, None, **lk_params)
				good_new = p1[flow_status == 1]
				good_old = p0[flow_status == 1]

				for i,(new,old) in enumerate(zip(good_new,good_old)):
					a,b = new.ravel()
					c,d = old.ravel()

					for i in good_new:
						x,y = i.ravel()
						cv2.circle(frame,(x,y),3,255,-1)


				p0 = good_new.reshape(-1, 1, 2) 
				prev_gray = gray.copy()
			# features_prev = features_curr
			LastPosOccluded = gotOccluded



	# if the object is present on the conveyor, but it was not found in the current frame (occlusion):
	if obj_present and not obj_found:
		# predict position with kalman filter
		center_rectangle = (center_rectangle[0]-5,center_rectangle[1]+2)
		point1 = (point1[0]-5,point1[1]+2)
		point2 = (point2[0]-5,point2[1]+2)
	
	# if object reaches the end of the conveyor:
	if obj_present and point1[0]<=belt_x0:
		# prepare for next object (reset status and roi)
		obj_present = False
		point1 = point1_start
		point2 = point2_start
		center_rectangle = getRectangleCenter(point1, point2)

		# reset stored areas
		areas.clear()
		JustGotOccluded = False
		LastPosOccluded = False
		gotOccluded = False
	
	center_rectangle_prev = center_rectangle
	
	# %% SOME NICE VISUALIZATION
	
	# draw semi-transparent box for visualizing classification and tracking info
	vis_infobox_transparency = 0.3
	vis_infobox_width = 300
	vis_infobox = np.zeros((h,vis_infobox_width,3),dtype="uint8")
	vis_infobox = cv2.addWeighted(frame[0:h,0:vis_infobox_width,:], vis_infobox_transparency, vis_infobox, (1-vis_infobox_transparency), 1.0)
	frame[0:h,0:vis_infobox_width,:] = vis_infobox
	
	# draw classification information
	vis_classification_y = 35
	vis_obj_height = 100
	vis_object_y = vis_classification_y+25
	vis_object_x = 20
	cv2.putText(frame, "CLASSIFICATION:", (10,vis_classification_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,150,0), 1, cv2.LINE_AA)
	vis_obj_picture = imutils.resize(obj_picture, height=vis_obj_height)
	frame[vis_object_y:vis_object_y+vis_obj_height,vis_object_x:(vis_object_x+vis_obj_picture.shape[1])] = vis_obj_picture
	cv2.putText(frame, obj_type, (20,vis_classification_y+vis_obj_height+50), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(frame, "Cup: "+str(obj_type_hist["cup"]), (30,vis_classification_y+vis_obj_height+75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(frame, "Book: "+str(obj_type_hist["book"]), (30,vis_classification_y+vis_obj_height+95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
	cv2.putText(frame, "Box: "+str(obj_type_hist["box"]), (30,vis_classification_y+vis_obj_height+115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

	# prepare text and color depending on object status
	if obj_present and obj_found:
		roi_color = (0,255,0)
		roi_text = "Tracking object " + str(obj_count)
	elif obj_present and not obj_found:
		roi_color = (0,255,255)
		roi_text = "Lost object " + str(obj_count) + ". Predicting"
	elif not obj_present:
		roi_color = (0,0,255)
		roi_text = "Waiting object " + str(obj_count+1)
	
	# display roi rectangle and center
	cv2.rectangle(frame, point1, point2, roi_color,2)
	cv2.circle(frame, center_rectangle, 1, roi_color, 2)
	# write object status above object
	cv2.putText(frame, roi_text, (point1[0],point1[1]-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, roi_color, 1, cv2.LINE_AA)


	######## 3D estimation
	# start_time = time.time()
	# Point to match in the left frame coords
	mp_left_frame = center_rectangle[0], center_rectangle[1]

	# getting matched point in right frame coords. TO BE ORGANIZED.
	mp_right_frame = matchPoint(mp_left_frame, template_h, template_w, roi_h, roi_left_off, roi_right_off, frame, frame_right, gray, gray_right)

	# Showing matched point on the right frame
	cv2.circle(frame_right, (int(mp_right_frame[0]), int(mp_right_frame[1])), 1, roi_color, 2)
	# print("took (ms): ", 1000*(time.time() - start_time))

	homogenous_mp_3d = cv2.triangulatePoints(mtx_P_l, mtx_P_r, mp_left_frame, mp_right_frame)
	mp_3d = cv2.transpose(homogenous_mp_3d)
	mp_3d = cv2.convertPointsFromHomogeneous(mp_3d)
	numpy_horizontal_concat = np.concatenate((frame, frame_right), axis=1)

	######## Show Frames
	# Update text info before showing frame
	drawTrackingInformation(mp_3d)
	# show final result
	cv2.imshow("Final Project", frame)
	cv2.imshow("mask", mask_fg)

	scale = 0.6
	cv2.imshow('Numpy Horizontal Concat', cv2.resize(numpy_horizontal_concat,None,fx=scale,fy=scale))

	# q to exit, space to pause
	k = cv2.waitKey(1)
	if k == ord('q'): break
	if k == 32: cv2.waitKey()

cv2.destroyAllWindows()