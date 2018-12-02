from __future__ import print_function
import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "path to the image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)
cv2.waitKey(0)

canny = cv2.Canny(image, 30, 150)
cv2.imshow("canny", canny)
cv2.waitKey(0)
