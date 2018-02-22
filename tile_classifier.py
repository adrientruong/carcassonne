from pipeline import PipelineStep
import cv2
import numpy as np
from canny import *
import string
from utils import *
from collections import defaultdict

class TileClassifier(PipelineStep):
    def __init__(self):
        self.edge_detector = CannyEdgeDetector(75, 250)
        self.templates = self.generate_templates()

    def generate_histogram(self, angles, magnitudes, nbins = 9):
        histogram = np.zeros(nbins)
        angle_step = 360 / nbins
        for iy, ix in np.ndindex(angles.shape):
            angle = angles[iy, ix]
            magnitude = magnitudes[iy, ix]
            bin1 = int(np.ceil(float(angle) / angle_step) - 1)
            if (angle / angle_step) % 1 >= 0.5 or angle / angle_step % 1 == 0:
                bin2 = bin1 + 1
            else:
                bin2 = bin1 - 1

            center_angle1 = (angle_step * (bin1 + 1)) - (angle_step / 2)
            center_angle2 = (angle_step * (bin2 + 1)) - (angle_step / 2)

            def wraparound(bin_i):
                if bin_i < 0:
                    return nbins - 1
                elif bin_i >= nbins:
                    return 0
                else:
                    return bin_i
            bin1 = wraparound(bin1)
            bin2 = wraparound(bin2)
            histogram[bin1] += magnitude * float(np.abs(angle - center_angle2)) / angle_step
            histogram[bin2] += magnitude * float(np.abs(angle - center_angle1)) / angle_step
        return histogram

    def normalized_img(self, img):
        img = np.copy(img)
        img = img.astype('float32')
        mean = img.mean()
        std = img.std()
        img -= mean
        img /= std

        return img

    def normalized_cross_correlation(self, raw_tile, template):
        normalized_tile = self.normalized_img(raw_tile)
        normalized_template = self.normalized_img(template)

        return (normalized_tile * normalized_template).sum()

    def calculate_dominant_angle(self, img):
        dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        angles = np.arctan2(dy, dx)
        angles = np.rad2deg(angles)
        angles[angles<0] += 360
        magnitudes = np.sqrt(dx**2 + dy **2)
        bins = 36
        histogram = self.generate_histogram(angles, magnitudes, nbins=bins)
        dominant_angle = np.argmax(histogram) * (360/bins)
        return dominant_angle

    def generate_template(self, img):
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dominant_angle = self.calculate_dominant_angle(img)
        #img = self.edge_detector.process({'img': img})['edges']
        
        return (img, dominant_angle)

    def generate_templates(self):
        templates = []
        for letter in string.ascii_uppercase[:24]:
            name = 'data/tiles/' + letter + '.png'
            tile_img = cv2.imread(name)
            tile_img = cv2.blur(tile_img, (3, 3))
            t = self.generate_template(tile_img)
            cv2.imwrite('data/tiles/templates/' + letter + '.png', t[0])
            templates.append(t)
        return templates

    def classify_tile(self, tile):
        tile, tile_angle = self.generate_template(tile)

        #cv2.imshow('tile', tile)

        #print('tile_angle:', tile_angle)
        scores = []
        i = 0
        rotated_templates = []
        for template, template_angle in self.templates:
            max_score = 0


            # diff = round_nearest(tile_angle - template_angle, 90) / 90
            # if diff < 0:
            #     diff = 4 - np.abs(diff)
            # rotated_template = np.rot90(template, k=4-diff)

            #print('template_angle:', template_angle)
            #print('rotating template ccw:', diff)
            # cv2.imshow('template', template)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            winning_template = None
            for r in range(4):
                rotated_template = np.rot90(template, k=r)
                score = self.normalized_cross_correlation(tile, rotated_template)
                if score > max_score:
                    max_score = score
                    winning_template = rotated_template
            rotated_templates.append(winning_template)

            #print('template ', string.ascii_uppercase[i], 'max_score:', max_score)
            scores.append(max_score)
            i += 1

        #print('scores:', scores)
        scores = np.array(scores)
        label = np.argmax(scores)
        sorted_scores = np.argsort(scores)

        # cv2.imshow('winning template ' + string.ascii_uppercase[sorted_scores[-1]], rotated_templates[sorted_scores[-1]])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #print('scores:', np.sort(scores))
        #print('scores indexes:', np.argsort(scores))
        letter = string.ascii_uppercase[label]
        return letter

    def process(self, inputs, visualize=False):
        tiles = inputs['tiles']
        labels = [self.classify_tile(t) for t in tiles]

        outputs = {'tile_labels': labels}
        # if visualize:
        #     img_copy = np.copy(inputs['img'])
        #     draw_polar_lines(img_copy, lines)
        #     outputs['debug_img'] = img_copy

        return outputs
