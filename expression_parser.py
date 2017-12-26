import numpy as np
import cv2 as cv
import pickle
import sys

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

if(len(sys.argv) != 2):
	print("usage: python3 expression_parser.py <name of image>")
	sys.exit()
 
#reading an image
image = cv.imread(sys.argv[1],0)
height, width = image.shape

#convert to binary image
for i in range(0, height):
    for j in range(0, width):
    	if(image[i,j] > 100):
    		image[i,j] = 0
    	else:
    		image[i,j] = 255

#split image into 28x28 sub-images, each containing an operator or digit

#dilate image to ensure numbers/operators are connected
kernel = np.array([[255,255],
				[255,255]])
dilation = cv.dilate(image,kernel,iterations = 1)
segmented_image = np.copy(dilation)

#find a bounding box for each digit/operator
bounds = []
for col in range(0,width): #columns
	for row in range(0, height): #rows
		if(segmented_image[row,col] == 255):
			left,right,top,bottom = findBoundingBox(segmented_image,row,col,col,col,row,row)
			bounds.append((left,right,top,bottom))

#now retrieve a matrix of grayscale values from each bounding box
#also detect minus and ones, because the bounding box causes the image to be too zoomed in
print("Bounds for each digit:",bounds)
figures = []
model = pickle.load(open('svm.sav', 'rb'))
kernel = np.array([[255,255],
				[255,255]])
for index,bound in enumerate(bounds):

	bounded_fig = np.array(dilation[ bound[2]:bound[3], bound[0]:bound[1]])

	#if 60% or more of array is white, assume its a minus or 1
	total = 0
	white = 0
	for i in bounded_fig:
		for j in i:
			if(j == 255):
				white += 1
			total += 1
	frequency = white/total
	is_minus_or_one = False
	if(frequency >= 0.6):
		is_minus_or_one = True

	#if more rows than columns, its a 1. Otherwise its a minus
	if(is_minus_or_one):
		if(len(bounded_fig[:,0]) > len(bounded_fig[0,:])):
			figures.append('1')
		else:
			figures.append('-')
	else:
		#use svm to classify the other digits/+

		#pad digits with black background, so they look similar to those in MNIST
		rows = 20 + abs(bound[3]-bound[2])
		cols = 20 + abs(bound[1] - bound[0])
		padded_fig = np.empty([rows,cols])
		padded_fig[:,:] = 0
		padded_fig[10:(rows-10),10:(cols-10)] = bounded_fig
		#scale image to 28x28 (MNIST digits are 28x28)
		img_scaled = cv.resize(padded_fig,(28,28), interpolation = cv.INTER_AREA)
		img_scaled = cv.erode(img_scaled,kernel,iterations = 1)

		#get 784 length row of pixels
		pixels = []
		for i in img_scaled:
			for j in i:
				pixels.append(int(j))
		pixels = np.array(pixels)
		pixels = pixels.reshape(1, -1)
		pixels = pixels/255.0
		#pass this to svm model to detect digit
		digit = model.predict(pixels)
		figures.append(digit[0])

#combine adjacent digits to form numbers
appended_digits = []
number = ""
for item in figures:
	if((item != 10) and (item != '-')):
		number += str(int(item))
	else:
		appended_digits.append(number)
		if(item == 10):
			appended_digits.append('+')
		else:
			appended_digits.append(item)
		number = ""

appended_digits.append(number)

#evaluate expression
numbers = []
operators = []
for item in appended_digits:
	if((item == '+') or (item == '-')):
		operators.append(item)
	else:
		numbers.append(item)

print('\n')
for op in operators:
	operand1 = int(numbers.pop(0))
	operand2 = int(numbers.pop(0))
	if(op == '+'):
		print(operand1,'+',operand2)
		ans = operand1+operand2
		numbers.insert(0,ans)
	else:
		print(operand1,'-',operand2)
		ans = operand1-operand2
		numbers.insert(0,ans)

print("Answer:",numbers[0])