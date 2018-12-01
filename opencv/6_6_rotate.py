
import argparse
import numpy as np
import cv2

import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "path to the image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])

rotated = imutils.rotate(image, 45)
cv2.imshow("rotated", rotated);
cv2.waitKey(0)
