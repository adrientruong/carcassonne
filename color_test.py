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

lower_img = img_with_color(green_lower)
#lower_img[:, :] = (0, 255, 0)
upper_img = img_with_color(green_upper)

for value in range(255, -1, -1):
	range_img = img_range_with_color((0, 0, 0), (180, 255, 255), value)
	show_image(range_img)

for value in range(green_lower[2], green_upper[2]):
	range_img = img_range_with_color(green_lower, green_upper, value)
	show_image(range_img)

show_images([lower_img, upper_img])