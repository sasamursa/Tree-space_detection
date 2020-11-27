# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:41:41 2020

@author: Eren
"""

import numpy as np
import cv2 as cv
import math
import time

def odometry_x(focal_length,distance,pixel_length,magnitude,angle):
    a=abs(math.cos(angle))
    if(a<270 and a>90):
        a=-a
    x=-1*magnitude*pixel_length*distance/focal_length
    return x
def odometry_y(focal_length,distance,pixel_length,magnitude,angle):
    y=math.sin(angle)*magnitude*pixel_length*distance/focal_length
    return y
def findAverage(lst):
    v=0
    for i in len(lst):
        for j in len(lst[i]):
            v+=lst[i]
    #return v/len(lst)
    
cap = cv.VideoCapture("Data/Video/DJI_0407.mov")
ret, frame1 = cap.read()
frame1 = cv.resize(frame1,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
position_x=0
position_y=0
f=0
while(1):
    ret, frame2 = cap.read()
    frame2 = cv.resize(frame2,(640,480),fx=0,fy=0, interpolation = cv.INTER_CUBIC)
    if ret == False:
        break
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    f+=1
    if(f%1!=0):
        continue
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    #time.sleep(0.1)
    x=np.mean(mag)
    a=np.mean(ang)
    degree=(a*180)/np.pi
    position_x+=odometry_x(25,10000,0.0625,x,degree)
    position_y+=odometry_y(25,10000,0.0625,x,degree)
    print(position_x)
    print(position_y)
    hsv[...,0] = 180/np.pi/2*ang
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)
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