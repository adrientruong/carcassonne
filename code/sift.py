from pipeline import PipelineStep
import cv2
import numpy as np

class SiftMatcher(PipelineStep):
	def process(self, inputs, visualize=False):
		
		outputs = {'img': img, 'corners': dst, 'debug_img': img_copy}

		return outputs