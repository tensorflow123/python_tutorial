import argparse
import numpy as np
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "path to the image")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])

#  [y1:y2, x1:x2]
crop = image[30:120, 240:335]

cv2.imshow("crop", crop);
cv2.waitKey(0)

