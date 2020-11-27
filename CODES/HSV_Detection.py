import cv2 as cv
import numpy as np
img = cv.imread('Data/F4.jpg') # Importing Sample Test Image

scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image if it was too large
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)


cv.imshow('Image',img)  # Showing The Sample Test Image
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_range = np.array([42,70,00])  # Set the Lower range value of color in BGR
upper_range = np.array([80,255,250])   # Set the Upper range value of color in BGR
mask = cv.inRange(hsv,lower_range,upper_range) # Create a mask with range
result = cv.bitwise_and(img,img,mask = mask)  # Performing bitwise and operation with mask in img variable
cv.imshow('Image1',result) # Image after bitwise operation
cv.waitKey(0)
cv.destroyWindow('Image1')
cv.destroyWindow('Image')