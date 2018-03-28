from pipeline import PipelineStep
import cv2
import numpy as np
from utils import *

class HoughLineTransform(PipelineStep):
    def __init__(self, rho_step, theta_step, threshold, key='edges'):
        self.rho_step = rho_step
        self.theta_step = theta_step
        self.threshold = threshold
        self.key = key

    def process(self, inputs, visualize=False):
        edges = inputs[self.key]
        lines = cv2.HoughLines(edges, self.rho_step, self.theta_step, self.threshold)
        lines = np.squeeze(lines)

        outputs = {'lines': lines}
        if visualize:
            img_copy = np.copy(inputs['img'])
            draw_polar_lines(img_copy, lines)
            outputs['debug_img'] = img_copy

        return outputs

class HoughLineProbabilisticTransform(PipelineStep):
    def __init__(self, rho_step, theta_step, threshold, min_line_len, max_line_gap, key='edges'):
        self.rho_step = rho_step
        self.theta_step = theta_step
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.key = key

    def process(self, inputs, visualize=False):
        edges = inputs[self.key]

        lines = cv2.HoughLinesP(edges,
                                self.rho_step,
                                self.theta_step,
                                self.threshold,
                                self.min_line_len,
                                self.max_line_gap)
        if lines is not None:
            lines = np.squeeze(lines, axis=1)

        outputs = {'line_segments': lines}
        if visualize:
            img_copy = np.copy(inputs['img'])
            #draw_2point_lines(img_copy, lines)
            draw_2point_line_segments(img_copy, lines)
            outputs['debug_img'] = img_copy

        return outputs