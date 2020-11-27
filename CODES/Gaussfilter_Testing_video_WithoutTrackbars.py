import cv2
import numpy as np

## image test importing codes ##


def nothing(x):
    pass

cap = cv2.VideoCapture('Data/Video/DJI_0384.mov')
cv2.namedWindow("Trackbars")



while True:
    
    ret, img = cap.read()
    if ret == True:
        img = cv2.resize(img,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


        
        lower_range = np.array([24, 98,56])
        upper_range = np.array([41, 255,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        
        kernel = np.ones((8,8), 'uint8')
        
        mask = cv2.GaussianBlur(mask,(11,11),18)
        mask = cv2.dilate(mask, kernel)
        result = cv2.bitwise_and(img, img, mask=mask)
        
        (_,contours,hierarchy) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area >2500):
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