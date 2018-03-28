from pipeline import PipelineStep
import cv2
import numpy as np

class BoardKMeansIdentifier(PipelineStep):
	def __init__(self, K):
		self.K = K

	def process(self, inputs, visualize=False):
		img = inputs['img']
		#mask = inputs['edges']
		Z = img.reshape((-1,3))
		#Z = img[mask != 0]
		Z = np.float32(Z)

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		ret, labels, centers = cv2.kmeans(Z, self.K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

		# Now convert back into uint8, and make original image
		centers = np.uint8(centers)
		print(centers)
		#centers[:] = np.array([0, 0, 0])
		colors = np.array([[255, 0, 0],
						   [0, 255, 0],
						   [0, 0, 255],
						   [100, 100, 100],
						   [200, 200, 200],
						   [50, 75, 50],
						   [25, 90, 255]])
		#centers = colors[:centers.shape[0], :]
		#centers[:] = np.array([0, 0, 0])
		#centers[1] = np.array([255, 255, 255])

		#labels[:] = 0
		#labels[1] = 2
		#centers = col

		# not_grid_mask = np.copy(mask)
		# not_grid_mask[not_grid_mask != 0] = labels.flatten()
		# kernel = np.ones((5,5),np.uint8)
		# not_grid_mask = cv2.dilate(not_grid_mask, kernel, iterations=1)

		outputs = {'img': img, 'edges': None}

		if visualize:
			# debug_img = np.copy(img)
			# debug_img[not_grid_mask == 0] = np.array([0, 0, 0])
			# debug_img[not_grid_mask != 0] = np.array([255, 255, 255])
			# kernel = np.ones((1,1),np.uint8)
			# cv2.dilate(debug_img, kernel, iterations=1)
			# outputs['debug_img'] = debug_img
			outputs['debug_img'] = centers[labels.flatten()].reshape((img.shape))

		return outputs