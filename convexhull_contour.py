 ### convex hull outer contour ### 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
img = cv2.imread('./defected/leaf22_modified.jpg')
rows = img.shape[0]
cols = img.shape[1]
# rows = int((rows*40)/100)
# cols = int((cols*40)/100)
# img = cv2.resize(img, (cols,rows)) 

g = img[:,:,1]
kernel = np.ones((2,2),np.uint8)
eroded2 = cv2.erode(g,kernel,iterations = 3)
ret4,thresh4 = cv2.threshold(eroded2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

edges = cv2.Canny(thresh4,0,255)
ppp,contours,hier = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
filled = cv2.drawContours(thresh4,contours,-1,(255,255,255),15)

# filled = ~filled

result_cord= []
coord = []
for i1 in range(rows-1):
    for j1 in range(cols-1):
        pixel1 = filled[i1,j1]
        pixel2 =  filled[i1,j1+1]
        result = abs(pixel2 - pixel1)
        if result>0:
            coord = (j1+1,i1)
#             print(coord)
            result_cord.append(coord)
# print('previous',result_cord)      
row_white = []
col_white = []
whitecoordi = []
whitecoordi2 = []
for i11 in range(rows):
     for j11 in range(cols):
        white = filled[i11,j11]
        if (i11==0 and white>0) or (i11==rows-1 and white>0):
            whitecoordi = (j11,i11)
            row_white.append(whitecoordi)
        if (j11==0 and white>0) or (j11==cols-1 and white>0):
            whitecoordi2 = (j11,i11)
            col_white.append(whitecoordi2)


white_coordi = row_white + col_white + result_cord
white_coordi = np.array(white_coordi)

hull = cv2.convexHull(white_coordi)
hull_coord1 = []
hull_coord2 = []
for i4 in range(len(hull)):
    hull_coord = hull[i4]
    x4, y4 = hull_coord[0]
    hull_coord1 = (x4, y4)
    hull_coord2.append(hull_coord1)
hull_new =  row_white + col_white + hull_coord2 
hull_new = np.array(hull_new)

def last(n):  
    return n[m]   
def sort(tuples): 
    return sorted(tuples, key = last) 
m = 0
hull_sorted_pts = sort(hull_new)

hull_update_pts = hull_sorted_pts
def euclidean(coords):
    x0, y0 = hull_update_pts[0]
    x, y = coords
    return ((x-x0)**2 + (y-y0)**2)**0.5
hull_first_ele = []
for i2 in range(len(hull_update_pts)-1):
    candidates = hull_update_pts[1:]
#     print('SORTED\n',candidates)
    candidates.sort(key=euclidean)
    hull_update_pts = candidates  
    hull_first_ele.append(candidates[0])

hull_first_ele = np.array(hull_first_ele[1:])

def euclidean2(coords):
    x60, y60 = hull_first_ele[-1]
    x6, y6 = coords
    return ((x6-x60)**2 + (y6-y60)**2)**0.5
connect = sort(hull_first_ele[:100])
for i66 in range(len(connect)):
    connect.sort(key = euclidean2)
f0,f1 = connect[0] 
f2,f3 = hull_first_ele[-1]
# for i55 in range(len(hull_first_ele)-1):
#     x50,y50 = hull_first_ele[i55]
#     x51,y51 = hull_first_ele[i55+1]
#     lineThickness = 2
#     cv2.line(img, (x50,y50), (x51,y51), (0,0,255), lineThickness)
#     cv2.line(img, (f0,f1), (f2,f3), (0,0,255), lineThickness)
hull_first_ele2 =[]
hull_first_ele3 =[]
for i70 in range(len(hull_first_ele)):
    f2,f3 = hull_first_ele[i70]
    if (f2,f3)!=(f0,f1):
        hull_first_ele2 = (f2,f3)
        hull_first_ele3.append(hull_first_ele2)

# hull_first_ele3 = hull_first_ele3.insert(0,connect[0])
hull_first_ele3 = [(f0,f1)]+hull_first_ele3
hull_first_ele3 = np.array(hull_first_ele3)
print(hull_first_ele3)
cv2.drawContours(img, [hull_first_ele3],  -1, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow("OUTPUT2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()