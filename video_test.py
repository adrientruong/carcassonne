import numpy as np
import cv2
from collections import defaultdict
from pipeline import *
from canny import *
from hough import *
from blur import *
from find_intersections import *
from find_homography import *
from dilate import *
from resize import *
from tile_classifier import *
from find_tiles import *
from board_reconstructor import *

from timeit import default_timer as timer

class Tracker():
    def __init__(self):
        CANNY_WEAK_THRESHOLD = 125
        CANNY_STRONG_THRESHOLD = 250
        HOUGH_RHO_STEP = 1
        HOUGH_THETA_STEP = np.pi/180
        HOUGH_THRESHOLD = 50
        HOUGH_MIN_LINE_LEN = 30
        HOUGH_MAX_LINE_GAP = 30

        full_pipeline = Pipeline()
        full_pipeline.add_step(Resize(max_width=800))
        full_pipeline.add_step(Blur(15, 0.75))
        full_pipeline.add_step(CannyEdgeDetector(CANNY_WEAK_THRESHOLD, CANNY_STRONG_THRESHOLD))
        full_pipeline.add_step(Dilate('edges', 2))
        full_pipeline.add_step(HoughLineProbabilisticTransform(HOUGH_RHO_STEP, HOUGH_THETA_STEP, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP))
        full_pipeline.add_step(FindIntersections())
        full_pipeline.add_step(RANSACHomography())
        full_pipeline.add_step(FindTiles(padding=5))
        full_pipeline.add_step(TileClassifier())
        full_pipeline.add_step(ReconstructBoard())

        partial_pipeline = Pipeline()
        partial_pipeline.add_step(FindTiles(padding=5))
        partial_pipeline.add_step(TileClassifier())
        partial_pipeline.add_step(ReconstructBoard())

        self.full = full_pipeline
        self.partial = partial_pipeline

        self.prev_outputs = None
        self.last_full_run = 0

    def get_board(self, frame):
        if timer() - self.last_full_run > 30:
            print('Starting from scracth!')
            self.prev_outputs = None

        start = timer()
        if self.prev_outputs is None:
            outputs = self.full.run({'img': frame})
        else:
            outputs = self.partial.run(prev_outputs)
        end = timer()
        print('Time to run piepline:', end - start)
        if outputs is None:
            return None

        self.prev_outputs = outputs
        self.last_full_run = timer()

        return outputs['board_img']

cap = cv2.VideoCapture(0)
tracker = Tracker()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = frame[100:-100, 250:-250]

    board = tracker.get_board(frame)

    if board is not None:
        # Our operations on the frame come here
        # Display the resulting frame
        cv2.imshow('recognized', board)
    
    cv2.imshow('raw', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()