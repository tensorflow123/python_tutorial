from __future__ import print_function
import argparse
import numpy as np
import cv2

print("cv2, max of 255: {}".format(cv2.add(np.uint8([255]), np.uint8([1]))))
print("cv2, min of 0: {}".format(cv2.subtract(np.uint8([0]), np.uint8([1]))))

print("numpy, max of 255: {}".format(np.uint8([255]) + np.uint8([1])))
print("numpy, min of 0: {}".format(np.uint8([0] - np.uint8([1]))))

