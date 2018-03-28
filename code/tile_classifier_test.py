import cv2
import numpy as np
from utils import *
import string

def generate_features(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    left_edge = img_hsv[:, :20]
    right_edge = img_hsv[:, -20:]
    top_edge = img_hsv[:20, :]
    bottom_edge = img_hsv[-20:, :]
    center = img_hsv[20:-20, 20:-20, :]

    left_avg_hue = left_edge[:, :].mean()
    right_avg_hue = right_edge[:, :, 0].mean()
    top_avg_hue = top_edge[:, :, 0].mean()
    bottom_avg_hue = bottom_edge[:, :, 0].mean()
    center_avg_hue = center[:, :, 0].mean()

    features = np.array([left_avg_hue, right_avg_hue, top_avg_hue, bottom_avg_hue, center_avg_hue])
    
    return features

tile_template_correct = cv2.imread('data/tiles/P.png')
tile_template_random = cv2.imread('data/tiles/A.png')

img_tile = cv2.imread('data/tile_test1/1.png')
img_tile = np.rot90(img_tile, k=1)

img_tile_features = generate_features(img_tile)
tile_template_correct_features = generate_features(tile_template_correct)
tile_template_random_features = generate_features(tile_template_random)

for letter in string.ascii_uppercase[:24]:
    name = 'data/tiles/' + letter + '.png'
    template_img = cv2.imread(name)
    template_features = generate_features(template_img)
    distance = np.linalg.norm(template_features - img_tile_features)
    print('distance to ' + letter + ' : ', distance)

show_image(img_tile)
