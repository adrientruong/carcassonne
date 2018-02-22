from pipeline import PipelineStep
import cv2
import numpy as np

class HarrisCornerDetector(PipelineStep):
	def process(self, inputs, visualize=False):
		img = inputs['img']
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray, 2, 3, 0.04)
		
		outputs = {'img': img, 'corners': dst}
		
		if visualize:
			dst = cv2.dilate(dst, None)
			img_copy = np.copy(img)
			img_copy[dst>0.01*dst.max()] = [0, 0, 255]
			outputs['debug_img'] = img_copy

		return outputs