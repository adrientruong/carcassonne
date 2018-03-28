import cv2
from utils import *

green_lower = (30, 150, 100)
green_upper = (50, 255, 255)

def img_with_color(color):
	img = np.zeros((100, 100, 3), dtype=np.uint8)
	img[:, :] = color

	img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

	return img

def img_range_with_color(lower, upper, value):
	img = np.zeros((upper[0] - lower[0], upper[1] - lower[1], 3), dtype=np.uint8)
	for i, h in enumerate(range(lower[0], upper[0])):
		for j, s in enumerate(range(lower[1], upper[1])):
			img[i, j] = (h, s, value)

	img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

	return img

lower_blue = (0, 0, 0)
upper_blue = (255, 90, 90)
for v in range(50, 90):
	range_img = img_range_with_color(lower_blue, upper_blue, v)
	show_image(range_img)