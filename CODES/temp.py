import cv2 as cv
import numpy as np
img = cv.imread('Data/images6.jpg') # Importing Sample Test Image
cv.imshow('Image',img)  # Showing The Sample Test Image
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_range = np.array([35,20,20])  # Set the Lower range value of color in BGR
upper_range = np.array([100,255,255])   # Set the Upper range value of color in BGR
mask = cv.inRange(hsv,lower_range,upper_range) # Create a mask with range
result = cv.bitwise_and(img,img,mask = mask)  # Performing bitwise and operation with mask in img variable
cv.imshow('Image1',result) # Image after bitwise operation
cv.waitKey(0)
cv.destroyWindow('Image1')
cv.destroyWindow('Image')