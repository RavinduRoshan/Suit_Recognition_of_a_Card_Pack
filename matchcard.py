# python matchcard.py --templates templates --images Cards

import sys
import numpy as np
import argparse
import glob
import cv2

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image

	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)

	else:
		r = width / float(w)
		dim = (width, int(h * r))

	resized = cv2.resize(image, dim, interpolation = inter)

	return resized

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--templates", required=True, help="Path to template images")
ap.add_argument("-i", "--images", required=True,
	help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

for image in glob.glob(args["images"] + "/*.jpg"):
	print("Loaded input image as " + image.split("\\", 1)[1].split(".")[0])
	image = cv2.imread(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	i=0
	value = [None]*4
	tmp = [None]*4

	for templatePath in glob.glob(args["templates"] + "/*.jpg"):
		print("Matching the template for " + templatePath.split("\\", 1)[1].split(".")[0] + ".....")
	
		template = cv2.imread(templatePath)
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		template = cv2.Canny(template, 50, 200)
		(tH, tW) = template.shape[:2]

		found = None

		for scale in np.linspace(0.2, 1.0, 20)[::-1]:

			resized = resize(gray, width = int(gray.shape[1] * scale))
			r = gray.shape[1] / float(resized.shape[1])

			if resized.shape[0] < tH or resized.shape[1] < tW:
				break

			edged = cv2.Canny(resized, 50, 200)
			result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
			#print(result)
			(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

			if args.get("visualize", False):
				clone = np.dstack([edged, edged, edged])
				cv2.waitKey(0)

			if found is None or maxVal > found[0]:
				found = (maxVal, maxLoc, r)
				
		(_, maxLoc, r) = found
		(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
		(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.imshow("Image", image)

		value[i] = found[0]
		tmp[i] = templatePath
		i += 1

	print("Template matching finished !.")
	pic = tmp[value.index(max(value))]
	print(value)
	shape = cv2.imread(pic)
	nm = pic.split("\\", 1)[1].split(".")[0]
	cv2.imshow("Input Shape", shape)
	print("Input card is a " + nm)
	cv2.waitKey(0)
