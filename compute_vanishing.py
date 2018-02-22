from pipeline import PipelineStep
import cv2
import numpy as np
from utils import *

class HoughLineTransform(PipelineStep):
    def __init__(self):
        self.rho_step = rho_step
        self.theta_step = theta_step
        self.threshold = threshold

    def process(self, inputs, visualize=False):
        lines1 = inputs['lines1']
        lines2 = inputs['lines2']

        lines1 = convert_to_vector_lines(lines1)
        lines2 = convert_to_vector_lines(lines2)



        outputs = {'lines': lines}
        if visualize:
            img_copy = np.copy(inputs['img'])
            draw_polar_lines(img_copy, lines)
            outputs['debug_img'] = img_copy

        return outputs

def convert_to_vector_lines(lines):
    vector_lines = []
    for line in lines:
        vector_lines.append(convert_to_vector_line(line))
    return np.array(vector_lines)

def convert_to_vector_line(line):
    point1 = np.array([line[0], line[1], 1])
    point2 = np.array([line[2], line[3], 1])

    return np.cross(point1, point2)