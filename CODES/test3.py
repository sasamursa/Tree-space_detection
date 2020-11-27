# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:41:41 2020

@author: Eren
"""

import numpy as np
import cv2 as cv
import math
import time

    
cap = cv.VideoCapture("Data/Video/DJI_0370.mov")
ret, frame1 = cap.read()
frame1 = cv.resize(frame1,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255


f=0
while(1):
    ret, frame2 = cap.read()
    frame2 = cv.resize(frame2,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
    if ret == False:
        break
    
     # Set trackbars
 
    
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    f+=1
    if(f%1!=0):
         continue
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 20, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    
    hsv[...,0] = 0
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_BGR2GRAY)
    
    bgr = cv.cvtColor(hsv,cv.COLOR_BGR2GRAY)
    bgrMean = bgr.mean()
    rgb_filtered = cv.inRange(bgr, bgrMean, 255)
    cutten = cv.bitwise_and(frame2, frame2, mask = rgb_filtered)
    '''
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv.KMEANS_RANDOM_CENTERS
    # Apply KMeans
    compactness,labels,centers = cv.kmeans(bgr,2,None,criteria,10,flags)
    print(compactness)
    print(labels)
    print(centers)
    #cutten = cv.bitwise_and(frame2, frame2, mask = rgb_filtered)
    '''
    Z = bgr.reshape((-1,1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    K = 2 
    _,labels,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    #A = np.all(Z[label.ravel()]==0])
    #B = np.all(Z[label.ravel()]==1])
    
    # flt = cv.normalize(label,(bgr.shape),0,255,cv.NORM_L1)
    #filt = np.all(flt == [0])
    #frame2[filt] = [255,255,255] 
    res = center[labels.flatten()]
    res2 = res.reshape((bgr.shape))
    label= labels.reshape((bgr.shape))
    masked_image = np.copy(frame2)
    
    if (Z[labels.ravel()==0].mean()) > (Z[labels.ravel()==1].mean()):
        masked_image[label == 1] = [0,0,0]    
    
    else:
        masked_image[label==0] = [0,0,0]
    # print(type(flt),type(bgr))
    # result = cv.bitwise_and(bgr,flt)
    #res_filtered = cv.inRange(res2, , 255)
    #kmeansFilter = cv.inRange(bgr,res,255)
    #result = cv.bitwise_and(frame2, frame2, mask= re)
    #cv.imshow('frame2',result)
    #cv.imshow('res',result)
    
    
    
    
    
    
    cv.imshow('frame',frame2)
    cv.imshow('Kmeans',masked_image)
    cv.imshow('meanMethod',cutten)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    if (f%2==0):
        prvs = next
cap.release()
cv.destroyAllWindows()