import numpy as np
import cv2 as cv

print("Reading data...")
data = np.genfromtxt('data.csv',delimiter=',')

image = np.empty([28,28])
for row in data[:,1:]:
	for i in range(0,28):
		for j in range(0,28):
			image[i,j] = row[(28*i)+j]
	cv.imshow('image',image)
	cv.waitKey(0)
