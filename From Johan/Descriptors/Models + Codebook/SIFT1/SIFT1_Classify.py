

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import savetxt
import pickle
import scipy.cluster.vq as vq
from numpy import histogram
from glob import glob
from os.path import exists, isdir, basename, join, splitext


codebook = pickle.load(open("/home/jo/PycharmProjects/Perception/desc/Models + Codebook/SIFT1/SIFT_(contrastThreshold=0.01,edgeThreshold=10)_Codebook.out", 'rb'))
clf = pickle.load(open('/home/jo/PycharmProjects/Perception/desc/Models + Codebook/SIFT1/SIFT_(contrastThreshold=0.01,edgeThreshold=10)_SVM_model.out', 'rb'))

#desc= cv2.xfeatures2d.SURF_create(600) #TJEK THREDSHOLD !!!!
desc = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.01,edgeThreshold=20) #,contrastThreshold=0.01,edgeThreshold=20

path =  "/home/jo/PycharmProjects/Perception/No_backround/object3.png"
#path = "/home/jo/Downloads/"
#path = "/home/jo/PycharmProjects/Perception/No_backround"
#path = "/home/jo/PycharmProjects/Perception/cropped_obj"


def createDescriptorToClassify(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_sift, des_sift = desc.detectAndCompute(gray, None)

    only_object = cv2.drawKeypoints(frame, kp_sift, frame,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('image', only_object)
    # cv2.imshow('image2',img2)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()

    print(len(des_sift))
    return des_sift


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    A = codebook.shape[0] + 1
    histogram_of_words, bin_edges = histogram(code,bins=range(A))
    return histogram_of_words

def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname)) for fname in glob(path + "/*.*", recursive=True)])
    return all_files

def sharpening(frame):
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])

    # applying the sharpening kernel to the input image & displaying it.
    #
    return cv2.filter2D(frame, -1, kernel_sharpening)

"""
def detectAndClassify():
    img_files = get_imgfiles(path)
    for i in img_files:

        img = cv2.imread(i)
        # img2 = cv2.imread(path2,0)
        #cv2.imshow('image', img)
        # cv2.imshow('image2',img2)
        #cv2.waitKey(1500)
        #cv2.destroyAllWindows()
        descriptors = createDescriptorToClassify(img)
        histogram = computeHistograms(codebook, descriptors)
        histogram = histogram.reshape(1, -1)
        # Classisy descriptor
        A = int(clf.predict(histogram))
        if A == 0:
            print("BOOK")
        elif A == 1:
            print("BOX")
        elif A == 2:
            print("CUP")
"""

def detectAndClassify(path):


    img = cv2.imread(path)
    # img2 = cv2.imread(path2,0)
    #cv2.imshow('image', img)
    # cv2.imshow('image2',img2)
    #cv2.waitKey(1500)
    #cv2.destroyAllWindows()
    descriptors = createDescriptorToClassify(img)
    histogram = computeHistograms(codebook, descriptors)
    histogram = histogram.reshape(1, -1)
    # Classisy descriptor
    A = int(clf.predict(histogram))
    if A == 0:
        print("BOOK")
    elif A == 1:
        print("BOX")
    elif A == 2:
        print("CUP")


detectAndClassify(path)

"""



#print(type(codebook))
#print(type(descriptors))
#print(codebook.shape[0])
#print(codebook.shape[1])
histogram = computeHistograms(codebook, descriptors)




#Descriptor tranformed to histogram, through the codebook
#X_histogram = computeHistograms(codebook, descripors)
histogram = histogram.reshape(1,-1)

#print(X_histogram.shape[1])


#(print("SÅÅÅÅÅÅ"))
#Classisy descriptor
A = int(clf.predict(histogram))
if A == 0:
    print("BOOK")
elif A == 1:
    print("BOX")
elif A== 2:
    print("CUP")

#Predict testset
"""