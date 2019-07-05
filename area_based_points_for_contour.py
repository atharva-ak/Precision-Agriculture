### 2nd Checkpoint and AREA #####
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

def last(n):  
    return n[m]   
   
# function to sort the tuple    
def sort(tuples): 
    return sorted(tuples, key = last) 

m = 0
white_coordi =np.array(white_coordi)
sorted_pts = sort(white_coordi)

update_pts = sorted_pts
def euclidean(coords):
    x0, y0 = update_pts[0]
    x, y = coords
    return ((x-x0)**2 + (y-y0)**2)**0.5
first_ele = []
for i2 in range(len(update_pts)-1):
    candidates = update_pts[1:]
    candidates.sort(key=euclidean)
    update_pts = candidates
#     print('SORTED\n',candidates)
    first_ele.append(candidates[0])

# first_ele.insert(0,sorted_pts[0])
# print('\n NEW\n',first_ele)

# print('\nsorted\n',first_ele) 
values = []
central_coordi = []
#### Angle ####
for i5 in range(len(first_ele)-2):
    x5,y5 = first_ele[i5]
    x15,y15 = first_ele[i5+1]
    x25,y25 = first_ele[i5+2]
    angle2 = np.arctan((y25 - y15)/(x25 - x15 +0.00005)) - np.arctan((y15-y5)/(x15-x5 +0.00005))
    angle2 = np.degrees(angle2)
    if angle2 < 0:
        angle2 = angle2 + 180.0
    values.append(angle2)
    central_coordi.append(first_ele[i5+1])
#     print('Angle',angle2)
#     print('Coordinate',first_ele[i5+1])
  
img1 = img.copy()
img2 = img.copy()
MM=[]
angle_val = []
for i6 in range(len(values)-1):
    val1 = values[i6]
    val2 = values[i6+1]
    val3 = abs(val2-val1)
    X1,Y1 = central_coordi[i6]
    X2,Y2 = central_coordi[i6+1]
    val4 = abs(X2-X1)
    M = (val3+0.00005)/(val4+0.00005)
    MM.append(M)
    angle_val.append(val3)
#     print('Measure',M)
#     print('Coordinate',central_coordi[i6])

medi1 = np.mean(MM)
# print('Mean',medi1)
coordd = []
modfies_coord = []
for i9 in range(len(MM)):
    if MM[i9]>medi1*4.5:
            X1,Y1 = central_coordi[i9]
            coordd = (X1,Y1)
            modfies_coord.append(coordd)
#             cv2.circle(img1,(X1,Y1),2,255,-1)
coordd2 = []
modfied_coord2 = []
for i10 in range(len(modfies_coord)-1):
    x10,y10 = modfies_coord[i10]
    x100,y100 = modfies_coord[i10+1]

    distance = ((x100-x10)**2 + (y100-y10)**2)**0.5
    if distance > 6:
#         print('Distance',distance)
#         cv2.circle(img1,(x10,y10 ),2,255,-1)
        coordd2 = (x10,y10)
        modfied_coord2.append(coordd2)
coordd3 = []
modfied_coord3 = []
for i20 in range(len(modfied_coord2)-2):
    x20,y20 = modfied_coord2[i20]
    x21,y21 = modfied_coord2[i20+1]
    x22,y22 = modfied_coord2[i20+2]
    angle21 = np.arctan((y22 - y21)/(x22 - x21 +0.00005)) - np.arctan((y21-y20)/(x21-x20 +0.00005))
    angle21 = np.degrees(angle21)
    if angle21 < 0:
        angle21 = angle21 + 180.0
    if angle21>70:
        cv2.circle(img2,(x21,y21),2,255,-1)
#         print('Angle_final',angle21)
        coordd3 = (x21,y21)
        modfied_coord3.append(coordd3)
img3 =img.copy()     
area_coord = []
area_coord2 = []
for i30 in range(len(modfied_coord3)-2):
    x30,y30 = modfied_coord3[i30]
    x31,y31 = modfied_coord3[i30+1]
    x32,y32 = modfied_coord3[i30+2]
    area = abs((x30*(y31-y32)+x31*(y32-y30)+x32*(y30-y31))/2)
#     print('Area',area)
    if area > 50:
        cv2.circle(img1,(x31,y31),2,255,-1)
        area_coord = (x31,y31)
        area_coord2.append(area_coord)
#         print('coord',(x31,y31))

area_coord3 = row_white + col_white + area_coord2
area_coord3 =np.array(area_coord3)

area_sorted_pts = sort(area_coord3)

area_update_pts = area_sorted_pts
def euclidean4(coords):
    x04, y04 = area_update_pts[0]
    x044, y044 = coords
    return ((x044-x04)**2 + (y044-y04)**2)**0.5
area_first_ele = []
for i80 in range(len(area_update_pts)-1):
    area_candidates = area_update_pts[1:]
#     print('SORTED\n',candidates)
    area_candidates.sort(key=euclidean4)
    area_update_pts = area_candidates  
    area_first_ele.append(area_candidates[0])

area_first_ele = np.array(area_first_ele[1:])

print(area_first_ele)



cv2.drawContours(img1, [area_first_ele],  -1, (0, 0, 255), 4, cv2.LINE_AA)
####### Second order differen ####            
# second_ord11 =[]
# print(len(MM))
# for i7 in range(len(MM)-1):
#     second_ord = abs(MM[i7+1]-MM[i7])
#     X11,Y11 = central_coordi[i7]
#     X22,Y22 = central_coordi[i7+1]
#     val5 = abs(X22-X11)
#     second_ord1 = second_ord/(val5+0.00005)
#     second_ord11.append(second_ord1)
# #     print('Second Ord:',second_ord1)
# #     
# medi = np.mean(second_ord11)
# print('Mean',medi)
# for i8 in range(len(second_ord11)):
#     if second_ord11[i8]>medi*4.5:
#             XX1,YY1 = central_coordi[i8]
#             cv2.circle(img2,(XX1,YY1),2,255,-1)
##########################


cv2.imshow("Output:middle", img1)

# cv2.imshow("masked", mask1)
# cv2.imshow("masked_G-channel", mask2)
cv2.waitKey(0)
cv2.destroyAllWindows()