import cv2 as cv
import numpy as np
img = cv.imread('Data/images3.jpg') # Importing Sample Test Image
cv.imshow('Image',img)  # Showing The Sample Test Image


lower_range = np.array([120,50,120])  # Set the Lower range value of color in BGR
upper_range = np.array([170,205,170])   # Set the Upper range value of color in BGR
mask = cv.inRange(img,lower_range,upper_range) # Create a mask with range
result = cv.bitwise_and(img,img,mask = mask)  # Performing bitwise and operation with mask in img variable
cv.imshow('Image1',result) # Image after bitwise operation
cv.waitKey(0)
cv.destroyWindow('Image1')
cv.destroyWindow('Image')