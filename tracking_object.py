import numpy as np
import cv2
import glob
import imutils

#%% LOAD IMAGES
dataset = '../Final_Project_GitHub/datasets/conveyor_without_occlusions'
images_left = glob.glob(dataset + '/left/*.png')
images_right = glob.glob(dataset + '/right/*.png')

assert images_right, images_left
assert (len(images_right) == len(images_left))
n_images = len(images_right)
images_right.sort()
images_left.sort()

#%% LOAD MATRICES AND PARAMETERS

dir = "../Final_Project_GitHub/calibration/calibration_matrix/"
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
img_sample = cv2.imread(images_left[0])
frame_height,  frame_width = img_sample.shape[:2]
object_detection = cv2.VideoWriter('Object_Detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 40, (frame_width,frame_height))

min_area = 3500

#Define kernels
kernel_ero = np.ones((20,20), np.uint8)
kernel_dil = np.ones((20,20), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

#Initialize Blackground Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

for i in range(n_images):
    #grab the current frame 
    frame = cv2.imread(images_left[i])

    # Apply the blackground subtration with the current frame
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Erode and dilate the mask
    erosion = cv2.erode(fgmask, kernel_ero, iterations=2)
    dilation = cv2.dilate(erosion, kernel_dil, iterations=1)

    # Find the contours
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
                
        if area > min_area:           
            # compute the bounding box for the contour, draw it on the frame
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (int(x-10),int(y-10)), (int(x+w+10), int(y+h+10)), (0,0,255),2)
                
    object_detection.write(frame)
    cv2.namedWindow("Blackground Subtration", cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Blackground Subtration", frame)
    cv2.imshow("Dilation", dilation)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
object_detection.release()
cv2.destroyAllWindows()