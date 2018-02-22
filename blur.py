from pipeline import PipelineStep
import cv2

class Blur(PipelineStep):
	def __init__(self, kernel_size, std):
		self.kernel_size = kernel_size
		self.std = std

	def process(self, inputs, visualize=False):
		img = inputs['img']
		blur = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.std)

		outputs = {'img': blur, 'debug_img': blur}

		return outputs