import numpy as np
import cv2
import glob
import imutils
import os

#%% FUNCTIONS USED

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

#%% LOAD MATRICES AND PARAMETERS

dir = "../calibration/calibration_matrix/"
mtx_l = np.load(dir + "camera_matrix_l.npy")
dist_l = np.load(dir + "distortion_coeff_l.npy")
mtx_l_new = np.load(dir + "camera_matrix_l_new.npy")
mtx_r = np.load(dir + "camera_matrix_r.npy")
dist_r = np.load(dir + "distortion_coeff_r.npy")
mtx_r_new = np.load(dir + "camera_matrix_r_new.npy")
mtx_R = np.load(dir + "rotation_l_r.npy")
mtx_T = np.load(dir + "translation_l_r.npy")
mtx_E = np.load(dir + "essential_matrix.npy")
mtx_F = np.load(dir + "fundamental_matrix.npy")
mtx_P_l = np.load(dir + "projection_matrix_l.npy")
mtx_P_r = np.load(dir + "projection_matrix_r.npy")

#%% MOTION DETECTION AND BACKGROUND MASK

# Create VideoWriter object in oder to record the result
#img_sample = cv2.imread(images_left[0])
#frame_height,  frame_width = img_sample.shape[:2]
#object_detection = cv2.VideoWriter('Object_Detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 40, (frame_width,frame_height))

min_area = 3500

#Define kernels
kernel_ero = np.ones((5,5), np.uint8)
kernel_dil = np.ones((20,20), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

#Initialize Blackground Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

#Define previous image
prev_img = cv2.imread(images_left[0])
prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(prev_img)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 100, qualityLevel = 0.1, minDistance = 10, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (7,7), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  

# Define region of the belt
contour_belt = np.array([[[387,476]],[[464,696]],[[1217,359]],[[1071,280]]])

object_on_conveyor = None
center_rectangle = (0,0)
new_object = False
object_number = 0
color = [(255,0,0),(0,255,0),(0,0,255),(100,100,100),(0,255,255),(255,0,255),(255,255,0)]

for i in range(n_images):
    # grab the current frame 
    frame = cv2.imread(images_left[i])
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply the blackground subtration with the current frame
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find the contours
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
                
        if area > min_area:           
            # compute the bounding box for the contour, draw it on the frame
            (x,y,w,h) = cv2.boundingRect(c)
            point1 = (int(x),int(y))
            point2 = (int(x+w), int(y+h))
            center_rectangle = centerRectangle_find(point1, point2)
            object_on_conveyor = cv2.pointPolygonTest(contour_belt, center_rectangle, measureDist = False)
            
            if object_on_conveyor==1.0 and new_object==False:
                new_object = True

            if object_on_conveyor==-1.0 and new_object==True:
                new_object = False
                object_number+=1
                mask = np.zeros_like(prev_img)         
                                            
            if object_on_conveyor==1.0:
                cv2.rectangle(frame, point1, point2, (0,0,255),2)
                cv2.circle(frame, center_rectangle, 1, (0,0,255), 2)

    output = cv2.bitwise_or(gray, gray, mask = fgmask)
    output = cv2.erode(output, kernel, iterations=1)
    output = cv2.dilate(output, kernel, iterations=1)


    if object_on_conveyor==1.0 and new_object==True:
        #GoodefeaturesToTrack()    
        prev_feat = cv2.goodFeaturesToTrack(prev_gray, mask = output, **feature_params)
        prev_feat = np.float32(prev_feat)

        for item in prev_feat:
            x, y = item[0]
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

        next_feat, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_feat, None, **lk_params)

        # Selects good feature points for previous position
        good_old = prev_feat[status == 1]
        # Selects good feature points for next position
        good_new = next_feat[status == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv2.line(mask, (a, b), (c, d), color[object_number], 1)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            img = cv2.circle(img, (a, b), 1, color[object_number], -1)

        # Overlays the optical flow tracks on the original frame
        result_opticalFlow = cv2.add(img, mask)
        prev_gray = gray.copy()
        prev_feat = good_new.reshape(-1,1,2)
        cv2.imshow("Features tracked",result_opticalFlow)

    os.system('clear')
    print("Object number: " + str(object_number))
    object_on_conveyor==None
    
    #object_detection.write(frame)
    cv2.namedWindow("Object tracked", cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Object tracked", frame)
    #cv2.imshow("Blackground Subtration", fgmask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

#object_detection.release()
cv2.destroyAllWindows()