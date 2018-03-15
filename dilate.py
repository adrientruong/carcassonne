from pipeline import PipelineStep
import cv2
import numpy as np

class Dilate(PipelineStep):
	def __init__(self, input_key, amount, output_key=None):
		self.input_key = input_key
		self.amount = amount
		self.output_key = output_key or input_key 

	def process(self, inputs, visualize=False):
		to_dilate = inputs[self.input_key]

		kernel = np.ones((self.amount, self.amount), np.uint8)
		dilated = cv2.dilate(to_dilate, kernel, iterations=1)

		outputs = {self.output_key: dilated}

		if visualize:
			outputs['debug_img'] = dilated

		return outputs

