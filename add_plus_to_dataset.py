import numpy as np
import cv2 as cv

def findBoundingBox(image,row,col,leftMost,rightMost,topMost,bottomMost):

	#left neighbor
	if(image[row,col-1] == 255):
		if(col-1 < leftMost):
			leftMost = col-1
		image[row,col] = 0
		leftMost,rightMost,topMost,bottomMost = findBoundingBox(image,row,col-1,leftMost,rightMost,topMost,bottomMost)

	#right neighbor
	if(image[row,col+1] == 255):
		if(col+1 > rightMost):
			rightMost = col+1
		image[row,col] = 0
		leftMost,rightMost,topMost,bottomMost = findBoundingBox(image,row,col+1,leftMost,rightMost,topMost,bottomMost)

	#top neighbor
	if(image[row-1,col] == 255):
		if(row-1 < topMost):
			topMost = row-1
		image[row,col] = 0
		leftMost,rightMost,topMost,bottomMost = findBoundingBox(image,row-1,col,leftMost,rightMost,topMost,bottomMost)

	#bottom neighbor
	if(image[row+1,col] == 255):
		if(row+1 > bottomMost):
			bottomMost = row+1
		image[row,col] = 0
		leftMost,rightMost,topMost,bottomMost = findBoundingBox(image,row+1,col,leftMost,rightMost,topMost,bottomMost)

	image[row,col] = 0;

	return leftMost,rightMost,topMost,bottomMost
 
print("Reading data...")
data = np.genfromtxt('data.csv',delimiter=',')
data = data[1:,:]
data = data.astype(np.int)
image = cv.imread('plus_signs.jpg',0)
height, width = image.shape

#convert to binary image
for i in range(0, height):
    for j in range(0, width):
    	if(image[i,j] > 100):
    		image[i,j] = 0
    	else:
    		image[i,j] = 255

#split image into 28x28 sub-images, each containing a plus sign

kernel = np.array([[255,255],
				[255,255]])
dilation = cv.dilate(image,kernel,iterations = 3)
segmented_image = np.copy(dilation)
dilation_image = np.copy(dilation)

#find a bounding box for each digit/operator
bounds = []
for col in range(0,width): #columns
	for row in range(0, height): #rows
		if(segmented_image[row,col] == 255):
			#cv.imshow('image',segmented_image)
			#cv.waitKey(0)
			left,right,top,bottom = findBoundingBox(segmented_image,row,col,col,col,row,row)
			bounds.append((left,right,top,bottom))

grid = []
for index,bound in enumerate(bounds):

	bounded_fig = np.array(dilation_image[ bound[2]:bound[3], bound[0]:bound[1]])
	#cv.imshow('image',bounded_fig)
	#cv.waitKey(0)

	#pad digits with black background, so they look similar to those in MNIST
	rows = 20 + abs(bound[3]-bound[2])
	cols = 20 + abs(bound[1] - bound[0])
	padded_fig = np.empty([rows,cols])
	padded_fig[:,:] = 0
	padded_fig[10:(rows-10),10:(cols-10)] = bounded_fig

	#scale image to 28x28 (MNIST digits are 28x28)
	img_scaled = cv.resize(padded_fig,(28,28), interpolation = cv.INTER_AREA)
	img_scaled = cv.erode(img_scaled,kernel,iterations = 1)
	#cv.imshow('image',img_scaled)
	#cv.waitKey(0)
	pixels = [10]
	for i in img_scaled:
		for j in i:
			pixels.append(int(j))
	#pixels = np.array(pixels)
	#pixels = pixels.reshape(1, -1)
	grid.append(pixels)
	#np.vstack([data, pixels])

for row in grid:
	data = np.vstack([data,row])

np.savetxt("data_with_plus.csv", data, delimiter=",", fmt='%i')