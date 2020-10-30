import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import color
from skimage.feature.texture import greycoprops,greycomatrix


def hsvHistogramFeatures(img):

	rows, cols, numOfBands = img.shape[:]
	# Convert the RGB image to HSV Color Space
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	# Extract the 3 components
	h = img[:,:,0]
	s = img[:,:,1]
	v = img[:,:,2]
	numberOfLevelsForH = 8 
	numberOfLevelsForS = 2 
	numberOfLevelsForV = 2
	# Calculate the maximum value of each component
	maxValueForH = np.max(h)
	maxValueForS = np.max(s)
	maxValueForV = np.max(v)
	# Create a 3D matrix that will store the value of the Histogram 
	hsvColor_Histogram = np.zeros((8, 2, 2))
	quantizedValueForH = np.ceil( h.dot(numberOfLevelsForH) / maxValueForH)
	quantizedValueForS = np.ceil( s.dot(numberOfLevelsForS) / maxValueForS)
	quantizedValueForV = np.ceil( v.dot(numberOfLevelsForV) / maxValueForV)
	index = np.zeros((rows*cols, 3))
	index[:,0] = quantizedValueForH.reshape(1,-1).reshape(1,quantizedValueForH.shape[0] * quantizedValueForH.shape[1]) 
	index[:,1] = quantizedValueForS.reshape(1,-1).reshape(1,quantizedValueForS.shape[0] * quantizedValueForS.shape[1]) 
	index[:,2] = quantizedValueForV.reshape(1,-1).reshape(1,quantizedValueForV.shape[0] * quantizedValueForV.shape[1])
	k=0

	for row in range(len(index[:,0])):
		if index[row,0] == 0 or index[row,1] == 0 or index[row,2] == 0:
			k+=1
			continue
		hsvColor_Histogram[int(index[row,0])-1,int(index[row,1])-1,int(index[row,2])-1] = hsvColor_Histogram[int(index[row,0])-1,int(index[row,1])-1,int(index[row,2])-1] + 1

	hsvColor_Histogram = hsvColor_Histogram[:].reshape(1,-1)
	# Normalize the Histogram
	hsvColor_Histogram = hsvColor_Histogram/np.sum(hsvColor_Histogram)
	# Reshape it to become 1*32
	return hsvColor_Histogram.reshape(-1)

def extractColorFeature(img):
	"""
	img : image in RGB that we will extract from it the mean and the std
	"""
	R = img[:,:,0]
	G = img[:,:,1]
	B = img[:,:,2]
	features = [np.mean(R),np.std(R),np.mean(G),np.std(G),np.mean(B),np.std(B)]
	features = features / np.mean(features)
	return features


def textureFeatures(img):
	img = color.rgb2gray(img)

	# Convert it to unsigned int to avoid problems of indexing
	img = skimage.img_as_ubyte(img)
	# Compute the greycomatrix
	glcm = greycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
	feature = greycoprops(glcm, 'dissimilarity')[0]
	feature = np.concatenate([feature,greycoprops(glcm, 'correlation')[0]])
	feature = np.concatenate([feature,greycoprops(glcm, 'contrast')[0]])
	feature = np.concatenate([feature,greycoprops(glcm, 'energy')[0]])
	feature = feature/np.sum(feature)
	#print(feature)
	return feature

def shapeFeatures(img):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	_,img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
	feature = cv2.HuMoments(cv2.moments(img))
	return feature.reshape(-1)


def getFeatures(img,featuresSize):
	if featuresSize >= 7 :
		features = extractColorFeature(img)
	if featuresSize >= 39 :
		features = np.concatenate([features, hsvHistogramFeatures(img)])
	if featuresSize >= 43:
		features = np.concatenate([features, textureFeatures(img)])
	if featuresSize >= 50:
		features = np.concatenate([features, shapeFeatures(img)])
	#print(features)
	return features




if __name__ == '__main__':
	main()