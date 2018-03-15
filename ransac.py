# from pipeline import PipelineStep
# import cv2
# import numpy as np
# from utils import *

# class RANSAC(PipelineStep):
#     def __init__(self, p):
#         self.p = p

#     def N(self):
#         return np.log(1-self.p) / np.log(1 - (1 - self.e))

#     def process(self, inputs, visualize=False):
#         edges = inputs['edges']
#         lines = cv2.HoughLines(edges, self.rho_step, self.theta_step, self.threshold)
#         lines = np.squeeze(lines)

#         outputs = {'lines': lines}
#         if visualize:
#             img_copy = np.copy(inputs['img'])
#             draw_polar_lines(img_copy, lines)
#             outputs['debug_img'] = img_copy

#         return outputs
