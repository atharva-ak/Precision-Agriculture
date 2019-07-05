### Centroid for stem ####

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
img = cv2.imread('leaf20.jpg')
rows = img.shape[0]
cols = img.shape[1]
rows = int((rows*40)/100)
cols = int((cols*40)/100)
img = cv2.resize(img, (cols,rows)) 

g = img[:,:,1]
kernel = np.ones((2,2),np.uint8)
eroded2 = cv2.erode(g,kernel,iterations = 3)
ret4,thresh4 = cv2.threshold(eroded2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

edges = cv2.Canny(thresh4,0,255)
ppp,contours,hier = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
filled = cv2.drawContours(thresh4,contours,-1,(255,255,255),15)
(winW, winH) = (cols, 35)
stepSize = 10
blob = []
centroid = []
for height in range(0, rows, stepSize):
    crop = filled[height:height + winH, 0:0 + winW]
    M = cv2.moments(crop)
    cv2.imshow("OUTPUT1", crop)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    blob = (cX, height+cY)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    centroid.append(blob)
centers = np.int0(centroid)
for center in centers:
    x,y = center.ravel()
    cv2.circle(img,(x,y),3,(0,0,255),-1)
cv2.imshow("OUTPUT2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()