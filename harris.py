from pipeline import PipelineStep
import cv2
import numpy as np

class HarrisCornerDetector(PipelineStep):
    def process(self, inputs, visualize=False):
        img = inputs['edges']
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray = np.float32(gray)
        dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        outputs = {'img': inputs['img'], 'corners': dst}
        
        if visualize:
            dst = cv2.dilate(dst, None)
            img_copy = np.copy(img)
            img_copy[dst>0] = 255
            outputs['debug_img'] = img_copy

        return outputs

class BetterCornerDetector(PipelineStep):
    def process(self, inputs, visualize=False):
        img = inputs['img']
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        gray = inputs['edges']

        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=30)
        corners = np.int0(corners)
        outputs = {'img': img, 'corners': corners}
        
        if visualize:
            img_copy = np.copy(img)
            for i in corners:
                x, y = i.ravel()
                cv2.circle(img_copy, (x, y), 3, 255, -1)

            outputs['debug_img'] = img_copy

        return outputs