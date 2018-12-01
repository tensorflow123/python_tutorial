import argparse
import numpy as np
import cv2

import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "path to the image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])

resized = imutils.resize(image, 50)
cv2.imshow("resized", resized);
cv2.waitKey(0)
