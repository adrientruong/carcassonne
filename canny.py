from pipeline import PipelineStep
import cv2
import numpy as np

class CannyEdgeDetector(PipelineStep):
	def __init__(self, low, high):
		self.low = low
		self.high = high

	def process(self, inputs, visualize=False):
		img = inputs['img']
		edges = cv2.Canny(img, self.low, self.high)
		
		outputs = {'edges': edges, 'debug_img': np.stack((edges,) * 3, -1)}

		return outputs