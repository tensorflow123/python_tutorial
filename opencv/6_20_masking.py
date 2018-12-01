from __future__ import print_function
import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "path to the image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)
cv2.waitKey(0)

# mask shoule be same size as image
mask_canvas = np.zeros(image.shape[:2], dtype = "uint8")
cv2.rectangle(mask_canvas, (25, 25), (275, 275), 255, -1)
cv2.imshow("mask_canvas", mask_canvas)
cv2.waitKey(0)

masked = cv2.bitwise_and(image, image, mask = mask_canvas)
cv2.imshow("masked", masked)
cv2.waitKey(0)
