import cv2
import numpy as np
from matplotlib import pyplot as plt




#-------------------IMAGE 1: Cells-------------------
#---PART 1---显示
#read image.
img = cv2.imread('1.jpg')

#display original image.
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


#---PART 2---二值化
#grayscale conversion.
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#binary conversion.
ret, img_bin = cv2.threshold(img_gray,127,255, 0)

#display binary image.
plt.figure()
plt.imshow(img_bin, cmap='gray')


#---PART 3---轮廓选取
#remove salt&pepper noise using median filter.
img_med = cv2.medianBlur(img_bin, 15)

#find boundary by Contour map
contours, hierarchy = cv2.findContours(img_med,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#display boundary image.

img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))




#---PART 4---轮廓面积,计数
#initialize a dict structure to save areas.
Areas={}
#Calculate areas and add to dict.
for i in range(1, len(contours), 1):
	Areas[str(i)] = cv2.contourArea(contours[i])

#draw "number:area" pairs on image
plt.figure()
plt.imshow(img)
for key, value in Areas.items():
	[xd,yd] = contours[int(key)][1][0]
	plt.text(xd, yd, key+':'+str(value), color="red", \
		fontdict={"fontsize":6,"fontweight":'bold',"ha":"left", "va":"baseline"})

#display number of cells.
print('There are ' + str(len(Areas)) + ' cells in total.')
print()





#-------------------IMAGE 2: lines-------------------
#Hough line detection
img = cv2.imread('2.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 50, 200)
lines = cv2.HoughLines(img_edges,1,np.pi/180,100, 10)

#draw line 1
for rho,theta in lines[0]:
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

#draw line 2
for rho,theta in lines[2]:
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

#calculate included angle
diff_angle = lines[0][0][1] - lines[2][0][1]

#radiant to degree
diff_angle = diff_angle*180/np.pi

#rounding
diff_angle = np.round(diff_angle,1)

#print result
print('The incluede angle between two lines is ' + str(diff_angle) + ' degree')
print()

plt.figure()
plt.imshow(img)




#-------------------IMAGE 3: circle center-------------------
#read and preprocess image
img = cv2.imread('3.jpg', 0)
img_blur = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#Hough circle detection
circles = cv2.HoughCircles(img_blur,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

#rounding
circles = np.uint16(np.around(circles))

#draw circles and centers.
for i in circles[0,:]:
	# draw the outer circle
	cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
	# draw the center of the circle
	cv2.circle(cimg,(i[0],i[1]),2,(255,0,0),3)
plt.figure()
plt.imshow(cimg)

#calculates distance.
x1 = circles[0,0,0]
y1 = circles[0,0,1]
x2 = circles[0,1,0]
y2 = circles[0,1,1]

distance = np.round(np.sqrt((x1-x2)**2+(y1-y2)**2),2)

#print center coordinates and distance between them.
print("The circle center coordinates are:")
print('('+str(x1)+','+str(y1)+')')
print('('+str(x2)+','+str(y2)+')')
print()
print("The distance between circle centers is:")
print(distance)
print()





#------------show result images---------------------
plt.show()
