from itertools import product

from board_img_translator import *

def draw_translator(img, img_translator):
	for x, y in product(range(-10, 10), range(-10, 10)):
		image_point = img_translator.board_p_to_img_p((x, y))
		cv2.circle(img, center=tuple(image_point), radius=2, thickness=2, color=(0, 0, 255))
		cv2.putText(img, '({}, {})'.format(x, y), org=tuple(image_point), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))