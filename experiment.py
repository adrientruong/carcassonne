import cv2
import numpy as np
import string
from canny import *
from tile_classifier import *

edge_detector = CannyEdgeDetector(125, 250)

tile = cv2.imread('tiles/A.png')
gray_tile = cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY)
#gray_tile = cv2.blur(gray_tile, (4, 4))
gray_tile = cv2.resize(gray_tile, (64, 64))
#gray_tile -= np.min(gray_tile)

board = cv2.imread('tiles/monastery_occluded.png')
#gray_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
gray_board = cv2.resize(board, (64, 64))
#gray_board = gray_board.astype(np.uint64)
#gray_board -= np.min(gray_board)
gray_board_edges = edge_detector.process({'img': gray_board})['edges']


# sift = cv2.xfeatures2d.SIFT_create()
# kp_tile, des_tile = sift.detectAndCompute(gray_tile, None)
# kp_board, des_board = sift.detectAndCompute(gray_board, None)

# bf = cv2.BFMatcher()
# matches = bf.match(des_tile, des_board)

# img = cv2.drawMatches(gray_tile, kp_tile, gray_board, kp_board, matches[:20], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# cv2.imshow('sift', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

classifier = TileClassifier()
raw_tile = cv2.imread('tiles/cityshield.png')
tiles = [raw_tile]
labels = classifier.process({'tiles': tiles})['tile_labels']
print(labels)