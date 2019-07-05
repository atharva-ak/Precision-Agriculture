### 1st Checkpoint #####
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
img = cv2.imread('./defected/leaf22_modified.jpg')
rows = img.shape[0]
cols = img.shape[1]
# rows = int((rows*15)/100)
# cols = int((cols*15)/100)
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

# print('ROW\n',row_white)
# print('COL\n',col_white)
white_coordi = row_white + col_white + result_cord
# corners = np.int0(white_coordi)
# for corner in corners:
#     x,y = corner.ravel()
#     cv2.circle(img,(x,y),3,255,-1)
    
# print('COL\n',white_coordi)
def last(n):  
    return n[m]   
   
# function to sort the tuple    
def sort(tuples): 
    return sorted(tuples, key = last) 

m = 0
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

first_ele.insert(0,sorted_pts[0])
# print('\n NEW\n',first_ele)

print('\nsorted\n',len(first_ele)) 
img111 = img.copy()
img222 = img.copy()

for i2 in range(len(first_ele)-1):
    x0,y0 = first_ele[i2]
    x1,y1 = first_ele[i2+1]
    lineThickness = 2
    cv2.line(img111, (x0, y0), (x1,y1), (0,0,255), lineThickness)
    distance = ((x1-x0)**2 + (y1-y0)**2)**0.5
    if distance <= 120:
        lineThickness = 2
        cv2.line(img222, (x0, y0), (x1,y1), (0,0,255), lineThickness)
   
   ## Mask ##
mask1 = cv2.bitwise_or(img, img, mask=filled)

first_ele = np.array(first_ele)

mask2 = mask1[:,:,1]

   ### PIXEL DENSITY THROUGH K MEANS ####
pixels = np.float32(mask2.reshape(-1, 1))
print(pixels.shape)
n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)

palette = np.uint8(palette)
res = palette[labels.flatten()]
res2 = res.reshape((mask2.shape))

green_pix = 0
other_pix = 0
for j14 in range(rows):
    for j24 in range(cols):
        pixel_val = res2[j14,j24]
        if pixel_val>= 1 and pixel_val<= 200:
            green_pix +=1
        elif pixel_val>200:
            other_pix +=1
print('Green:',green_pix)
print('Other:',other_pix)
        
# dominant = palette[np.argmax(counts)]
print('Dominant',counts)
total = green_pix+other_pix
percentage = (green_pix*100)/total
print(percentage)


# cv2.imwrite("./results/Green_channel.jpg", g)
# cv2.imwrite("./results/Noise_removal:- erosion,otsu-thresh,Canny_edge_detection.jpg", edges)
# cv2.imwrite("./results/contour_filling.jpg", filled)
# cv2.imwrite("./results/Output:boarder.jpg", img222)
# cv2.imshow("Green_channel", g)
# cv2.imshow("Noise_removal:- erosion,otsu-thresh,Canny_edge_detection", edges)
# cv2.imshow("results/contour_filling", filled)
cv2.imshow("Output:boarder", filled)
cv2.imshow("masked", mask1)
cv2.imshow("masked_G-channel", mask2)
cv2.waitKey(0)
cv2.destroyAllWindows()