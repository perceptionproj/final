import numpy as np
import cv2
import glob
import imutils

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
img_sample = cv2.imread(images_left[0])
frame_height,  frame_width = img_sample.shape[:2]
object_detection = cv2.VideoWriter('Object_Detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))
motion_detection = cv2.VideoWriter('Motion_Detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))

firstFrame = None
min_area = 4000

for i in range(n_images):
    #grab the current frame 
    frame = cv2.imread(images_left[i])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (27,27), 0)

    if firstFrame is None:
        firstFrame = gray
        continue
    
    # compute the absolute difference between the current frame and the first frame
    # delta = |background_model – current_frame|. In our case, background_model=firstFrame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 40, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, the find the contours on the thresholded image
    thresh = cv2.dilate(thresh, None, iterations=4)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        # if contour too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)

    cv2.putText(frame, "Object Detection", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frameDelta, "Delta Frame", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(thresh, "Thresh", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    object_detection.write(frame)
    thresh_video = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
    motion_detection.write(thresh_video)
    
    output = np.hstack((frameDelta, thresh))
    cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Motion Detection", 1300, 500)
    cv2.imshow("Motion Detection",output)
    cv2.imshow("Object Detection", frame)

	#cv2.imshow("Frame Delta", frameDelta)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

object_detection.release()
motion_detection.release()
cv2.destroyAllWindows()