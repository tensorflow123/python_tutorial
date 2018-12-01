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

(B, G, R) = cv2.split(image)

cv2.imshow("red", R)
cv2.imshow("green", G)
cv2.imshow("blue", B)
cv2.waitKey(0)


merged = cv2.merge([B, G, R])
cv2.imshow("merged", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
