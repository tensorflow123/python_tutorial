from __future__ import print_function
import argparse
import numpy as np
import cv2

rectangle = np.zeros((300, 300), dtype = "uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("rectangle", rectangle)
cv2.waitKey(0)

circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("circle", circle)
cv2.waitKey(0)

bitwise_and = cv2.bitwise_and(rectangle, circle)
cv2.imshow("and", bitwise_and)
cv2.waitKey(0)

bitwise_or = cv2.bitwise_or(rectangle, circle)
cv2.imshow("or", bitwise_or)
cv2.waitKey(0)

bitwise_xor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("xor", bitwise_xor)
cv2.waitKey(0)

bitwise_not = cv2.bitwise_not(rectangle, circle)
cv2.imshow("not", bitwise_not)
cv2.waitKey(0)
