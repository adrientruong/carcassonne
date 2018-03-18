import cv2
import numpy as np
from resize import *

class VideoFeed():
    def __init__(self, process, fps=10, num_frames_similar_threshold=3, diff_threshold=3000000):
        self.fps = fps
        self.num_frames_similar_threshold = num_frames_similar_threshold
        self.diff_threshold = diff_threshold

        self.process = process

        self.prev_frame = None
        self.num_similar_frames = 0
        self.num_frames = 0

        self.resizer = Resize(max_width=800)

    def diff_between_frames(self, frame1, frame2):
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame1, frame2).sum()

        return diff

    def start(self):
        cap = cv2.VideoCapture(0)
        while (True):
            ret, frame = cap.read()
            self.num_frames += 1
            if self.num_frames % self.fps == 0:
                self.num_frames = 0
            else:
                continue

            frame = frame[:, 500:-250]
            if self.prev_frame is not None:
                diff = self.diff_between_frames(frame, self.prev_frame)
                print('diff:', diff)
                if diff < self.diff_threshold:
                    self.num_similar_frames += 1
                else:
                    self.num_similar_frames = 0

            resized_frame = self.resizer.process({'img': frame})['img']
            cv2.imshow('raw', resized_frame)
            self.prev_frame = frame

            if self.num_similar_frames > self.num_frames_similar_threshold:
                print('similar frames!')
                self.process(frame)

                self.num_similar_frames = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
