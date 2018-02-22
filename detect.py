import numpy as np
import cv2
from collections import defaultdict
from pipeline import *
from correct_perspective import *
from canny import *
from hough import *
from line_filter import *
from kmeans import *
from harris import *
from blur import *
from compute_k_vanishing import *
from threshold import *
from line_classifier import *
from compute_vanishing import *

points = np.array([[167, 89],
                   [486, 80],
                   [523, 370],
                   [140, 380]], dtype='float32')
points2 = np.array([[155, 306],
                   [377, 114],
                   [652, 251],
                   [464, 514]], dtype='float32')
points3 = np.array([[240, 17],
                   [779, 128],
                   [681, 434],
                   [35, 271]], dtype='float32')

points4 = np.array([[304, 151],
                   [610, 155],
                   [700, 344],
                   [243, 329]], dtype='float32')

# cv2.imshow('top_down', board)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#board = correct_perspective(board, points2)
#detect_board(board)

# CANNY_WEAK_THRESHOLD = 125
# CANNY_STRONG_THRESHOLD = 250
CANNY_WEAK_THRESHOLD = 125
CANNY_STRONG_THRESHOLD = 250
HOUGH_RHO_STEP = 1
HOUGH_THETA_STEP = np.pi/180
HOUGH_THRESHOLD = 70
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 10

pipeline = Pipeline()
#pipeline.add_step(CorrectPerspective())
#pipeline.add_step(BoardKMeansIdentifier(2))
#pipeline.add_step(Blur(9, 1.5))
#pipeline.add_step(HarrisCornerDetector())
#pipeline.add_step(ThresholdHSV(np.array([30, 0, 0]), np.array([86, 255, 255])))
pipeline.add_step(ThresholdHSV(np.array([0, 0, 220]), np.array([255, 50, 255])))
#pipeline.add_step(HarrisCornerDetector())
#pipeline.add_step(CannyEdgeDetector(CANNY_WEAK_THRESHOLD, CANNY_STRONG_THRESHOLD))

#pipeline.add_step(BoardKMeansIdentifier(6))
#pipeline.add_step(HoughLineProbabilisticTransform(HOUGH_RHO_STEP, HOUGH_THETA_STEP, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP))
#pipeline.add_step(LineClassifier())
#pipeline.add_step(HoughLineTransform(HOUGH_RHO_STEP, HOUGH_THETA_STEP, HOUGH_THRESHOLD))
#pipeline.add_step(LineFilter())
#pipeline.add_step(ComputeK())

board = cv2.imread('data/tiles/X.png')
outputs = pipeline.run({'img': board, 'camera_points': points2}, visualize=True)

