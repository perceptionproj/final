#%% IMPORT
import numpy as np
import cv2
import glob
import imutils
import os
from sklearn.neighbors import NearestNeighbors

# %% FUNCTIONS
def getBlueMask(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	low_blue = np.array([105,40,95])
	up_blue = np.array([120,255,255])
	mask = cv2.inRange(hsv, low_blue, up_blue)
	kernel = np.ones((5,5),np.uint8)
	mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
	mask_morph = cv2.bitwise_not(mask_morph)	
	return mask_morph

def centerRectangle_find(p1, p2):
    x = int(p1[0] + (p2[0] - p1[0])/2)
    y = int(p1[1] + (p2[1] - p1[1])/2)
    return (x,y)

#%% LOAD IMAGES
dataset = '../datasets/conveyor_without_occlusions'
images_left = glob.glob(dataset + '/left/*.png')
images_right = glob.glob(dataset + '/right/*.png')

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

#%% SETTINGS and VARIABLES DEFINITION
h, w = cv2.imread(images_left[0]).shape[:2] # size of the images (pixels)
min_area = 3000 # define minimum area to avoid small outliers regions

# initialize frame_prev and features_prev
feat_prev = np.empty((0,1,2),dtype='float32')
gray_prev = np.zeros((h,w),dtype='uint8')

# Define region of the belt
contour_belt = np.array([[[387,476]],[[464,696]],[[1217,359]],[[1071,280]]])

#Define kernels
kernel_ero = np.ones((5,5), np.uint8)
kernel_dil = np.ones((20,20), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

#Initialize Blackground Subtractor
fgbg = cv2.createBackgroundSubtractorKNN(history=600,dist2Threshold=800, detectShadows=False)

#Define previous image
prev_img = cv2.imread(images_left[0])
prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 3000, qualityLevel = 0.01, minDistance = 1, blockSize = 2)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (7,7), maxLevel = 1, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  

of_ngb_dist_max = 300 # maximun medium distance to the feature's nearest neighbours (pixels)
min_N = 5 # minumin number of points that have to pass all the outlier tests in order for the measurement to be considered

object_on_conveyor = None
new_object = False
object_number = 0
center_rectangle = (0,0)
color = (0,255,0)

#%% TRACKING 2D

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

    # Apply the blackground subtration with the current frame
    fgmask = fgbg.apply(frame)

    # Get blue mask that characterizes the conveyor belt
    blue_mask = getBlueMask(frame)
    
	# Combine blue mask and blackground subtractor
    mask_object = cv2.bitwise_or(gray, gray, mask = fgmask)
    mask_object = cv2.erode(mask_object, kernel, iterations=1)
    mask_object = cv2.dilate(mask_object, kernel, iterations=1)
    mask_object = cv2.bitwise_and(mask_object, blue_mask)
    fgmask = cv2.morphologyEx(mask_object, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find the contours
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
                
        if area > min_area:           
            # compute the bounding box for the contour, draw it on the frame
            (x,y,width,height) = cv2.boundingRect(c)
            point1 = (int(x),int(y))
            point2 = (int(x+width), int(y+height))
            center_rectangle = centerRectangle_find(point1, point2)
            object_on_conveyor = cv2.pointPolygonTest(contour_belt, center_rectangle, measureDist = False)
            
            if object_on_conveyor==1.0 and new_object==False:
                new_object = True

            if object_on_conveyor==-1.0 and new_object==True:
                new_object = False
                object_number+=1
                                            
            if object_on_conveyor==1.0:
                cv2.rectangle(frame, point1, point2, (0,0,255),2)
                cv2.circle(frame, center_rectangle, 1, (0,0,255), 2)

    if object_on_conveyor==1.0 and new_object==True:
 
        prev_feat = cv2.goodFeaturesToTrack(prev_gray, mask = mask_object, **feature_params)
        prev_feat = np.float32(prev_feat)

        curr_feat, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_feat, None, **lk_params)
        
        # FIRST OUTLIER REMOVAL: only keep points with positive status and small error
        mask1 = np.logical_and(status==1,error<3).squeeze()

        prev_feat = prev_feat[mask1]
        curr_feat = curr_feat[mask1]

        # THIRD OUTLIER REMOVAL: remove points whose average distance to the closest neighbours is too big
        if (curr_feat.shape[0] >= min_N):
            nbrs = NearestNeighbors(n_neighbors=int(curr_feat.shape[0]/2), algorithm='ball_tree').fit(curr_feat[:,0,:])
            distances, _ = nbrs.kneighbors(curr_feat[:,0,:])
            distances_avg = distances[:,1:].mean(axis=1)
            mask3 = distances_avg < of_ngb_dist_max
            curr_feat = curr_feat[mask3]

        for i in range(curr_feat.shape[0]):			
            cv2.circle(frame, (curr_feat[i][0][0], curr_feat[i][0][1]), 2, color, -1)	

        prev_feat = curr_feat.reshape(-1,1,2)

    prev_gray = gray

    os.system("clear")
    print("Object Number: " + str(object_number))

    cv2.imshow("Object tracked", frame)
    #cv2.imshow("Blackground Subtration", mask_object)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()