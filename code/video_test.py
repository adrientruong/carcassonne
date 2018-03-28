import numpy as np
import cv2

from tracker import Tracker
from video_feed import VideoFeed

tracker = Tracker()
def process_frame(frame):
    outputs = tracker.process_frame(frame)
    # if outputs is not None:
    #     board = outputs['board_img']
    #     board_homography = outputs['board_homography']
    #     cv2.imshow('board homography', board_homography)
    #     cv2.imshow('board', board)

feed = VideoFeed(process=process_frame)
feed.start()

