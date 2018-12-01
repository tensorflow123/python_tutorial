import argparse
import numpy as np
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "path to the image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])

flipped = cv2.flip(image, 1)
cv2.imshow("flipped", flipped);
cv2.waitKey(0)

flipped = cv2.flip(image, 0)
cv2.imshow("flipped", flipped);
cv2.waitKey(0)

flipped = cv2.flip(image, -1)
cv2.imshow("flipped", flipped);
cv2.waitKey(0)
