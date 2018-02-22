from pipeline import PipelineStep
import cv2
import numpy as np

class CannyEdgeDetector(PipelineStep):
	def __init__(self, low, high):
		self.low = low
		self.high = high

	def process(self, inputs, visualize=False):
		img = inputs['img']
		#mask = inputs['labels']
		# v = np.median(img)
		# sigma = 1.5
		# lower = int(max(0, (1.0 - sigma) * v))
		# upper = int(min(255, (1.0 + sigma) * v))
		edges = cv2.Canny(img, self.low, self.high)
		#edges = cv2.Canny(img, lower, upper)

		#kernel = np.ones((2, 2), np.uint8)
		#edges = cv2.dilate(edges, kernel, iterations=1)

		#edges[mask != 2] = 0

		outputs = {'edges': edges, 'debug_img': np.stack((edges,) * 3, -1)}

		return outputs