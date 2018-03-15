import numpy as np
import cv2
from canny import *
from utils import *
img = cv2.imread('data/tiles/G.png')
#img = img[2:, 2:, :]
img = cv2.blur(img, (2, 2))
canny = CannyEdgeDetector(15, 30)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
edges = canny.process({'img': img_hsv[:, :, 0]})['edges']
# edges[:1, :] = 1
# edges[-1:, :] = 1
# edges[:, :1] = 1
# edges[:, :-1] = 1
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
good_contours = []
for c in contours:
	hull = cv2.convexHull(c)
	area = cv2.contourArea(c)
	print('area:', area)
	if area > 20:
		good_contours.append(hull)
cv2.drawContours(img, good_contours, -1, (0, 0, 255), 2)
cv2.imshow('tile', img)
cv2.waitKey(0)
cv2.destroyAllWindows()