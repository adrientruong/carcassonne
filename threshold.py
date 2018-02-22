from pipeline import PipelineStep
import cv2
import numpy as np

class ThresholdHSV(PipelineStep):
	def __init__(self, lower, upper):
		self.lower = lower
		self.upper = upper

	def process(self, inputs, visualize=False):
		img = inputs['img']
		img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(img_hsv, self.lower, self.upper)

		# corners = inputs['corners']
		# #corners = cv2.dilate(corners, None)
		# img[corners>0.01*corners.max()] = [0, 0, 255]
		
		result = cv2.bitwise_and(img, img, mask=mask)

		outputs = {'img': result, 'debug_img': result}

		return outputs