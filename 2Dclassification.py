#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:11:34 2020

@author: jkc
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def cutOutItem(testframe, empty_testframe):
    cuttedItem = 5
    return cuttedItem


# cv2.namedWindow('Window1')
cap = cv2.VideoCapture(
    '/Users/jkc/Documents/Uddannelse/DTU/Autonome Systemer/31392 - Perception for Autonnoumos Systems/Exam Project/Downloads/Stereo_conveyor_with_occlusions.mp4')
# cap.set(3,640)
# cap.set(4,480)


sift = cv2.xfeatures2d.SIFT_create(sigma=1)
orb = cv2.ORB_create()

#backSub = cv2.createBackgroundSubtractorMOG2()
backSub = cv2.bgsegm.createBackgroundSubtractorMOG()


ret, frame2 = cap.read()

start_point_border_rect = (680, 350)
end_point_border_rect = (718, 550)
color_border_rect = (255, 100, 0)
thickness_border_rect = 2

start_point_test_rect = (290, 350)
end_point_test_rect = (680, 700)
color_test_rect = (255, 255, 255)
thickness_test_rect = 2

framecount = 0
borderframe_total_sum = 0

change_factor = 1.05
object_reg = False
classify_images = []
classify_count = 0
empty_testframe_set = True

while ret:

    frame1 = frame2

    ret, frame2 = cap.read()



    # gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    # gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

    cv2.rectangle(frame2, start_point_border_rect, end_point_border_rect, color_border_rect, thickness_border_rect)
    cv2.rectangle(frame2, start_point_test_rect, end_point_test_rect, color_test_rect, thickness_test_rect)

    borderframe = frame1[350:550, 680:718]
    testframe1 = frame1[350:700, 290:680]

    fgMask = backSub.apply(testframe1)

    kp_sift, des_sift = sift.detectAndCompute(borderframe, None)

    print(len(kp_sift))

    #cv2.imshow('Frame', frame2)
    #cv2.imshow('FG Mask', fgMask)

    #cv2.imshow('image', frame1)
    cv2.waitKey(1)

    if len(kp_sift) > 2:
        object_reg = True
        print("object registreret")
        zerocount = [1, 1, 1, 1, 1]
        if empty_testframe_set == True:
            empty_testframe = testframe1
            empty_testframe_set = False

    if object_reg == True:
        zerocount.append(len(kp_sift))
        if (zerocount[len(zerocount) - 1] + zerocount[len(zerocount) - 2] + zerocount[len(zerocount) - 3] + zerocount[
            len(zerocount) - 4] + zerocount[len(zerocount) - 5]) == 0:
            print("tag test billede")
            only_object = cv2.bitwise_or(testframe1,testframe1,mask=fgMask)
            kp_sift_object, des_sift_object = sift.detectAndCompute(only_object, None)
            only_object = cv2.drawKeypoints(only_object, kp_sift_object,only_object, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.imshow('FG Mask', only_object)
            object_reg = False
            empty_testframe_set = True

# print(classify_count)

print(len(classify_images))
print(type(testframe1))
print(testframe1.shape)

cv2.destroyWindow('Window1')

