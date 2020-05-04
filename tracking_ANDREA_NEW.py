#%% IMPORT
import numpy as np
import cv2
import glob
import imutils

# %% LOAD CALIBRATION MATRICES AND CAMERA PARAMETERS
dir_calib = "../calibration/calibration_matrix/"
mtx_P_l = np.load(dir_calib + "projection_matrix_l.npy")
mtx_P_r = np.load(dir_calib + "projection_matrix_r.npy")
rect_map_l_x = np.load(dir_calib + "map_l_x.npy")
rect_map_l_y = np.load(dir_calib + "map_l_y.npy")
rect_map_r_x = np.load(dir_calib + "map_r_x.npy")
rect_map_r_y = np.load(dir_calib + "map_r_y.npy")

# %% LOAD IMAGES
dataset = '../datasets/conveyor_without_occlusions'
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

# %% SETTINGS and VARIABLES DEFINITION
h, w = cv2.imread(images_left[0]).shape[:2] # size of the images (pixels)
min_area = 3000 # define minimum object area to avoid small outliers regions (pixels^2)

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

# %% MAIN

for i in range(n_images):

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
	
	# combine blue mask and background subtractor
	mask_fg = cv2.bitwise_and(mask_fg, mask_belt_x)
	mask_fg = cv2.bitwise_and(mask_fg, mask_blue)
	mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=2)

	# find the contours
	cnts = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	obj_found = False

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
			
			# if the center didn't move too much
			if (np.linalg.norm(center_movement)<200):
				# check that object is on the conveyor
				object_on_conveyor = cv2.pointPolygonTest(belt_contour, center_rectangle, measureDist = False)
				if object_on_conveyor==1.0:
					if not obj_present:
						obj_count += 1
					obj_present = True
					obj_found = True
			else:
				point1 = point1_start
				point2 = point2_start
				center_rectangle = getRectangleCenter(point1, point2)

	
	# if the object is present on the conveyor, but it was not found in the current frame:
	if obj_present and not obj_found:
		# predict position with kalman filter
		center_rectangle = (center_rectangle[0]-4,center_rectangle[1]+2)
		point1 = (point1[0]-4,point1[1]+2)
		point2 = (point2[0]-4,point2[1]+2)
	
	# if object reaches the end of the conveyor:
	if obj_present and point1[0]<=belt_x0:
		# prepare for next object
		obj_present = False
		point1 = point1_start
		point2 = point2_start
		center_rectangle = getRectangleCenter(point1, point2)
	
	center_rectangle_prev = center_rectangle
	
	# %% VISUALIZATION
	
	cv2.rectangle(frame, point1, point2, (0,0,255),2)
	cv2.circle(frame, center_rectangle, 1, (0,0,255), 2)
	# write info above object
	cv2.putText(frame, str(obj_count), (point1[0],point1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
	cv2.imshow("Final Project", frame)
	#cv2.imshow("Blackground Subtration", mask_fg)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()