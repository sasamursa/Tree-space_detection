import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    __, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    lower_range = np.array([40,70,30])  # Set the Lower range value of color in BGR
    upper_range = np.array([100,255,250])
    
    mask = cv.inRange(hsv,lower_range,upper_range) 
    result = cv.bitwise_and(frame,frame,mask = mask) 
    
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('result', result)
    
    
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
cap.release()