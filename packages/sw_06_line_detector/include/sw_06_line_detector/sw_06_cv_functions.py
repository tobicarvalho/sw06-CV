import cv2
import numpy as np
import rospy
from numpy.lib.stride_tricks import as_strided

## Software Exercise 6: Choose your category (1 or 2) and replace the cv2 code by your own!

## CATEGORY 1
def inRange(hsv_image, low_range, high_range):
	output=np.ones((hsv_image.shape[0],hsv_image.shape[1]), dtype=np.uint8)
	for i in range(3):
		output &= ((hsv_image[:,:,i]>low_range[i]) & (hsv_image[:,:,i]<high_range[i]))
	return output
	#return cv2.inRange(hsv_image, low_range, high_range)

def bitwise_or(bitwise1, bitwise2):
	return bitwise1 | bitwise2
	#return cv2.bitwise_or(bitwise1, bitwise2)

def bitwise_and(bitwise1, bitwise2):
	# return bitwise1 & bitwise2
	
	return cv2.bitwise_and(bitwise1, bitwise2)

def getStructuringElement(shape, size):
	# Assuming we always want a ellipse kernel

	b=int(size[0]/2)
	a=int(size[1]/2)
	kernel=np.zeros(size,dtype=np.uint8)

	for i in range(size[0]):
		dy=i-b
		dx=int(round(a*np.sqrt((1-dy**2/b**2))))
		left=a-dx
		right=a+dx+1
		kernel[i,left:right]=1

	return kernel

	# return cv2.getStructuringElement(shape, size)

def dilate(bitwise, kernel):

	padding=(kernel.shape[0]-1)/2 # Assuming the kernel is odd
	
	bitwise=np.pad(bitwise,padding, 'constant')

	bitwise = as_strided(
	    bitwise,
	    shape=(
		bitwise.shape[0] - kernel.shape[0] + 1,  # The feature map is a few pixels smaller than the input
		bitwise.shape[1] - kernel.shape[1] + 1,
		kernel.shape[0],
		kernel.shape[1],
	    ),
	    strides=(
		bitwise.strides[0],
		bitwise.strides[1],
		bitwise.strides[0],  # When we move one step in the 3rd dimension, we should move one step in the original data too
		bitwise.strides[1],
	    )
	)

	bitwise=(bitwise*kernel).max(axis=(2,3))

	return bitwise

	# return cv2.dilate(bitwise, kernel)



## CATEGORY 2
def Canny(image, threshold1, threshold2, apertureSize=3):
	return cv2.Canny(image, threshold1, threshold2, apertureSize=3)


## CATEGORY 3 (This is a bonus!)
def HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap):
	return cv2.HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap)
