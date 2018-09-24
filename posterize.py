"""
Author      : Yi-Chieh Wu, Devon Frost
Class       : HMC CS 121
Date        : 2018 Sep 13
Description : Posterize image
"""

import os
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

######################################################################
# functions
######################################################################

### ========== TODO : START ========== ###
def posterize(img, k):
	#recast k as int
	k =int(k)

	# get the images width and height
	height, width = img.shape[:2]
	
	# reshape the image into vector so that k-means can be used
	image = img.reshape((img.shape[0] * img.shape[1], 3))
	
	# run k means
	cluster = KMeans(n_clusters=k, random_state=0).fit(image) 
	labels = cluster.predict(image) # predict k-cluster value of all pixels
	# create new image that assigns k-cluster info to each pixel
	newImg = cluster.cluster_centers_.astype("uint8")[labels] 

	# reshape posterized image
	newImg = newImg.reshape((height, width, 3))

	# return posterized image
	return newImg

def ImgComb (ogimg, posimg):
	# get the height and width of the images
	height, width, depth = ogimg.shape

	# create new image of double the width of the original
	comboImg = np.zeros((height, (width*2) , depth), np.uint8)

	#print (ogimg[0, 0])
	#print(ogimg[0, 0, 0])

	#assign original images pixels to the new image
	for y in range(height):
		for x in range(width):
			comboImg[y, x] = ogimg [y, x]

	#assign posterized images pixels to the new image
	for a in range(height):
		for b in range(width):
			comboImg[a, b+width] = posimg [a, b]

	return comboImg


### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main():
	### ========== TODO : START ========== ###
	#create results images folder
	results_dir = os.path.join("results")
	if not os.path.exists(results_dir):
		 os.mkdir(results_dir)
	
	# get user input
	imgFile = input("Image file name: ")
	k = input("Number of colors for posterized image: ")

	# read image from user 
	path = ""
	full_fn = os.path.join(path, imgFile)
	img = cv.imread(full_fn)
	# posterize image
	posImg = posterize(img, k)
	posName= (os.path.splitext(imgFile)[0]) + "Posterized.jpg" # creates filename
	cv.imwrite(os.path.join(results_dir, posName), posImg) # saves side by side file

	#creates image that has the original image and the posterized images side by side
	sideComp = ImgComb(img, posImg)
	compName= (os.path.splitext(imgFile)[0]) + "PosterizedComparison.jpg" # creates filename
	cv.imwrite(os.path.join(results_dir, compName), sideComp) # saves side by side file

	cv.imshow("side by side comparison", sideComp) # shows side by side comparison
	cv.waitKey(0) 
	### ========== TODO : END ========== ###

if __name__ == '__main__':
    main()
