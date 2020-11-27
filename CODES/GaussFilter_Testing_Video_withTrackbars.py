import cv2
import numpy as np

## image test importing codes ##


def nothing(x):
    pass

cap = cv2.VideoCapture('Data/Video/DJI_0384.mov')
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L  H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L  S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L  V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U  H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U  S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U  V", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Area", "Trackbars", 2000, 8000, nothing)
cv2.createTrackbar("filt", "Trackbars", 3, 10, nothing)
while True:
    
    ret, img = cap.read()
    if ret == True:
        img = cv2.resize(img,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L  H", "Trackbars")
        l_s = cv2.getTrackbarPos("L  S", "Trackbars")
        l_v = cv2.getTrackbarPos("L  V", "Trackbars")
        u_h = cv2.getTrackbarPos("U  H", "Trackbars")
        u_s = cv2.getTrackbarPos("U  S", "Trackbars")
        u_v = cv2.getTrackbarPos("U  V", "Trackbars")
        areaBar = cv2.getTrackbarPos("Area", "Trackbars")
        filt = cv2.getTrackbarPos("filt", "Trackbars")
        
        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((filt,filt), 'uint8')
        
        mask = cv2.GaussianBlur(mask,(9,9),21)
        mask = cv2.dilate(mask, kernel)
        result = cv2.bitwise_and(img, img, mask=mask)
        
        (_,contours,hierarchy) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area >areaBar):
                x,y,w,h = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)
                cv2.putText(img, 'Tree', (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255))
                 
        
        cv2.imshow("frame", img)
        cv2.imshow("mask", mask)
        cv2.imshow("result", result)
    
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()