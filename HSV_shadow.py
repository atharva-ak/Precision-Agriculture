#### HSV approach total contour detection ######

import cv2
import numpy as np

img = cv2.imread('./leaf_data/DJI_0195.JPG', -1)
rows = img.shape[0]
cols = img.shape[1]
rows = int((rows*40)/100)
cols = int((cols*40)/100)
img = cv2.resize(img, (cols,rows))
mask_contour = np.zeros((rows,cols), np.uint8)


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
red = hsv[:,:,0]
rettt,thresh7 = cv2.threshold(red,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('shadows_out11', thresh7)
remove_noise = cv2.medianBlur(thresh7,3)
remove_noise2 = cv2.medianBlur(remove_noise,3)
remove_noise3 = cv2.medianBlur(remove_noise2,5)

contours7, _ = cv2.findContours(thresh7,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
c = max(contours7, key = cv2.contourArea)
print(c)
result = cv2.drawContours(mask_contour, [c], 0, (255, 255, 255), -1, cv2.LINE_AA)

cv2.imshow("Result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()