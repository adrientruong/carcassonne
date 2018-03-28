from pipeline import PipelineStep
import cv2
import numpy as np
from kmeans import *

class LineClassifier(PipelineStep):
    def process(self, inputs, visualize=False):
        lines = inputs['lines']
        thetas = []
        for x1, y1, x2, y2 in lines:
            y = y2 - y1
            x = x2 - x1
            theta = np.arctan2(y, x)
            thetas.append(theta)
        labels = []

        thetas = np.array(thetas)
        thetas[thetas < 0] += np.deg2rad(180)

        for i, theta in np.ndenumerate(thetas):
            print(theta)
            if np.abs(theta) < np.deg2rad(5):
                labels.append(0)
            elif np.abs(np.abs(theta) - np.deg2rad(90)) < np.deg2rad(5):
                labels.append(1)
            else:
                labels.append(2)

        labels = np.array(labels)

        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # ret, labels, centers = cv2.kmeans(thetas, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        lines1 = lines[labels == 0]
        lines2 = lines[labels == 1]

        outputs = {'lines1': lines1, 'lines2': lines2}
        if visualize:
            img_copy = np.copy(inputs['img'])
            for x1, y1, x2, y2 in lines1:
                cv2.line(img_copy, (x1,y1), (x2,y2), (0, 255, 0), 2)
            for x1, y1, x2, y2 in lines2:
                cv2.line(img_copy, (x1,y1), (x2,y2), (0, 0, 255), 2)

            outputs['debug_img'] = img_copy

        return outputs