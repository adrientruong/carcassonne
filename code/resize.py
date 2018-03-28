from pipeline import PipelineStep
import cv2

class Resize(PipelineStep):
	def __init__(self, max_width):
		self.max_width = max_width

	def process(self, inputs, visualize=False):
		img = inputs['img']
		scale_factor = self.max_width / img.shape[1]
		resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
		
		outputs = {'img': resized, 'debug_img': resized}

		return outputs