import numpy as np
import cv2

class Pipeline:
	def __init__(self):
		self.steps = []

	def add_step(self, step):
		self.steps.append(step)

	def run(self, inputs, visualize=False):
		outputs = None
		imgs = [inputs['img']]
		for i, step in enumerate(self.steps):
			if i + 1 == len(self.steps):
				visualize = True
			outputs = step.process(inputs, visualize)

			if visualize:
				imgs.append(outputs['debug_img'])

			for key in outputs:
				inputs[key] = outputs[key]

		for i, img in enumerate(imgs):
			cv2.imshow('step ' + str(i), img)

		cv2.waitKey(0)
		cv2.destroyAllWindows()

		return outputs

class PipelineStep:
	def process(self, inputs, visualize=False):
		assert(False, "must override")