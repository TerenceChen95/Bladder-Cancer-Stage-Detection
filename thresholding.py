# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:02:59 2019

@author: tians
"""

import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
  
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def thresholding(img, origin_img, path):
    gray = rgb2gray(img).astype("uint8")
    arr = np.asarray(gray, dtype="uint8")
    for j in range(arr.shape[1]):
        for i in range(arr.shape[0]):
            if(arr[i][j]>=60 and arr[i][j]<=180):
                arr[i][j] = 255
            else:
                arr[i][j] = 0

    im2, contours, hierarchy = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    C = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area<10000 and area>1000:
            C.append(contour)
    #assume only 1 bbox detected
    location = []
    if(len(C) > 0):
        location = cv2.boundingRect(C[0])
        x, y, w, h = location
        plt.figure()
        rec = cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2)
        cv2.imwrite('./thresholding.png', rec)
        
        #plot in original img
        plt.figure()
        origin_img = cv2.resize(origin_img, (224, 224), interpolation=cv2.INTER_CUBIC)
        rec2 = cv2.rectangle(origin_img,(x,y),(x+w,y+h),(255, 0, 0),2)
        cv2.imwrite('./thresholding_origin.png', rec2)
        #plt.savefig(path+"thresholding.png", rec)

path = 'C:\\Users\\tians\\Desktop\\bladder cancer\\'
img_name = 'TCGA-ZF-AA5P40_bbox'
img = cv2.imread(img_name + '-resnet50-cam.jpg')
origin_img = cv2.imread(img_name+'.jpg')
thresholding(img, origin_img, path)
