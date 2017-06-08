# python matchnew.py --templates templates --images Cards

import sys
import numpy as np
import argparse
import glob
import cv2

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)

	else:
		r = width / float(w)
		dim = (width, int(h * r))

	resized = cv2.resize(image, dim, interpolation = inter)

	return resized

#def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--templates", required=True, help="Path to template images")
ap.add_argument("-i", "--images", required=True,
	help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

# load the input image, convert it to grayscale, and detect edges
for image in glob.glob(args["images"] + "/*.jpg"):
	print("Loaded input image as " + image.split("\\", 1)[1].split(".")[0])
	image = cv2.imread(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#cv2.waitKey(0)
	i=0
	value = [None]*4
	tmp = [None]*4

	# loop over the images to find the template in
	for templatePath in glob.glob(args["templates"] + "/*.jpg"):
		print("Matching the template for " + templatePath.split("\\", 1)[1].split(".")[0] + ".....")
		# load the template, convert it to grayscale, and initialize the
		# bookkeeping variable to keep track of the matched template
		template = cv2.imread(templatePath)
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		template = cv2.Canny(template, 50, 200)
		(tH, tW) = template.shape[:2]
		#cv2.imshow("Image", template)
		found = None

		for scale in np.linspace(0.2, 1.0, 20)[::-1]:
			#print(scale)
			# resize the image according to the scale, and keep track
			# of the ratio of the resizing
			resized = resize(gray, width = int(gray.shape[1] * scale))
			r = gray.shape[1] / float(resized.shape[1])

			# if the resized image is smaller than the template, then break
			# from the loop
			if resized.shape[0] < tH or resized.shape[1] < tW:
				break

			# detect edges in the resized, grayscale image and apply template
			# matching to find the template in the image
			edged = cv2.Canny(resized, 50, 200)
			#cv2.imshow("Pre processed", edged)
			result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

			# check to see if the iteration should be visualized
			if args.get("visualize", False):
				# draw a bounding box around the detected region
				clone = np.dstack([edged, edged, edged])
				#cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				#	(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
				#cv2.imshow("Visualize", clone)
				cv2.waitKey(0)

			# if we have found a new maximum correlation value, then ipdate
			# the bookkeeping variable
			if found is None or maxVal > found[0]:
				found = (maxVal, maxLoc, r)
				#print(found[1])
		# unpack the bookkeeping varaible and compute the (x, y) coordinates
		# of the bounding box based on the resized ratio
		(_, maxLoc, r) = found
		(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
		(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

		# draw a bounding box around the detected result and display the image
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.imshow("Image", image)
		#cv2.waitKey(0)

		value[i] = found[0]

		#print(i)
		tmp[i] = templatePath
		#print(tmp[i])
		i += 1
		#cv2.waitKey(0)
	print("Template matching finished !.")
	pic = tmp[value.index(max(value))]
	shape = cv2.imread(pic)
	nm = pic.split("\\", 1)[1].split(".")[0]
	cv2.imshow("Input Shape", shape)
	print("Input card is a " + nm)
	cv2.waitKey(0)
