import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mplt

#read image
img5 = cv2.imread('5.jpg')
img6 = cv2.imread('6.jpg')


#re-adjust image 5 to same shape as image 6
img5 = img5[1:,1:,:]


#histogram of figure 5
plt.figure(figsize=(12,12))
plt.subplot(231)
plt.imshow(img5[:,:,0], cmap=mplt.cm.Reds_r)
plt.subplot(232)
plt.imshow(img5[:,:,1], cmap=mplt.cm.Greens_r)
plt.subplot(233)
plt.imshow(img5[:,:,2], cmap=mplt.cm.Blues_r)
plt.subplot(234)
plt.hist(img5[:,:,0].flatten(), bins=range(257), align='left', color='gray')
plt.subplot(235)
plt.hist(img5[:,:,1].flatten(), bins=range(257), align='left', color='gray')
plt.subplot(236)
plt.hist(img5[:,:,2].flatten(), bins=range(257), align='left', color='gray')

#histogram of figure 6
plt.figure(figsize=(12,12))
plt.subplot(231)
plt.imshow(img6[:,:,0], cmap=mplt.cm.Reds_r)
plt.subplot(232)
plt.imshow(img6[:,:,1], cmap=mplt.cm.Greens_r)
plt.subplot(233)
plt.imshow(img6[:,:,2], cmap=mplt.cm.Blues_r)
plt.subplot(234)
plt.hist(img6[:,:,0].flatten(), bins=range(257), align='left', color='gray')
plt.subplot(235)
plt.hist(img6[:,:,1].flatten(), bins=range(257), align='left', color='gray')
plt.subplot(236)
plt.hist(img6[:,:,2].flatten(), bins=range(257), align='left', color='gray')

#display histogram
plt.show()

#correlation
cor_red		=	np.corrcoef(img5[:,:,0].flatten(), img6[:,:,0].flatten())
cor_green	=	np.corrcoef(img5[:,:,1].flatten(), img6[:,:,1].flatten())
cor_blue	=	np.corrcoef(img5[:,:,2].flatten(), img6[:,:,2].flatten())

#print correlation for rgb channel.
print("The correlation coefficient of red channel is: ",cor_red[0,1]) 
print("The correlation coefficient of green channel is: ",cor_green[0,1])
print("The correlation coefficient of blue channel is: ",cor_blue[0,1])
print()