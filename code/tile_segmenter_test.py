import numpy as np
import cv2

def segment_tile(tile):
	tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

	CITY_COLOR = np.array([41, 31, 77])
	GRASS_COLOR = np.array([101, 27, 79])
	ROAD_COLOR = np.array([195, 2, 98])
	MONASTERY_COLOR = np.array([7, 52, 73])
	FEATURE_COLORS = np.array([CITY_COLOR, GRASS_COLOR, ROAD_COLOR, MONASTERY_COLOR])

	print('tile:', tile)
	segmented_img = np.copy(tile)
	for y, x in np.ndindex(tile.shape[:2]):
		color = tile[y, x]
		min_distance = 100000
		closest_color = None
		for feature_color in FEATURE_COLORS:
			distance = np.linalg.norm(feature_color[0] - color[0])
			if distance < min_distance or closest_color is None:
				closest_color = feature_color
				min_distance = distance
		segmented_img[y, x] = closest_color
		#segmented_img[y, x] = np.array([101, 27, 79])


	#segmented_img[:, :] = np.array([101, 27, 79])
	print("hsv:", segmented_img)
	segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_HSV2BGR)
	print("bgr:", segmented_img)

	return segmented_img

tile = cv2.imread('data/tiles/F.png')
segmented_tile = segment_tile(tile)
cv2.imshow('tile', segmented_tile)
cv2.waitKey(0)
cv2.destroyAllWindows()