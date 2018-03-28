from pipeline import PipelineStep
import cv2
import numpy as np
from canny import *
import string
from utils import *
from collections import defaultdict
from scipy.spatial.distance import cdist

class SIFTTileClassifier(PipelineStep):
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.template_descriptors = self.generate_template_descriptors()

    def generate_template_descriptors(self):
        descriptors = []
        for letter in string.ascii_uppercase[:24]:
            name = 'data/tiles/' + letter + '.png'
            template = cv2.imread(name)
            template = cv2.blur(template, (3, 3))
            template = cv2.resize(template, (64, 64))
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            _, d = self.sift.detectAndCompute(template, None)

            descriptors.append((letter, template, d))
        return descriptors
            
    def match_keypoints(self, descriptors1, descriptors2, threshold = 0.7):
        distances = cdist(descriptors1, descriptors2)
        matches = []
        for i in range(descriptors1.shape[0]):
            distances_to_others = distances[i]
            partitioned = np.argpartition(distances_to_others, (1, 2))
            closest_distance = distances_to_others[partitioned[0]]
            second_closest_distance = distances_to_others[partitioned[1]]
            ratio = closest_distance / second_closest_distance
            if ratio <= threshold:
                matches.append([i, partitioned[0]])
        matches = np.array(matches)

        return matches

    def refine_match(self, keypoints1, keypoints2, matches, reprojection_threshold = 5,
        num_iterations = 1000):
        def convert_to_homo(points):
            return np.hstack([points, np.ones((points.shape[0], 1))])

        def calculate_reprojection_error(H, points1, points2):
            points1_homo = convert_to_homo(points1)
            projected_points = H.dot(points1_homo.T).T
            projected_points = projected_points[:, :2] / projected_points[:, 2].reshape((projected_points.shape[0], 1))
            error = np.linalg.norm(points2 - projected_points, axis=1)
            return error

        def find_homography(points1, points2):
            points1 = convert_to_homo(points1)
            points2 = convert_to_homo(points2)

            return np.linalg.lstsq(points1, points2)[0].T

        best_inlier_indices = None
        best_H = None
        matched_keypoints1 = keypoints1[matches[:, 0], :2]
        matched_keypoints2 = keypoints2[matches[:, 1], :2]
        for i in range(num_iterations):
            # Sample 4 points to get a hypothesis for H
            sample_indices = np.random.randint(0, matches.shape[0], size=4)
            samples = matches[sample_indices, :]
            sample_points1 = keypoints1[samples[:, 0]][:, :2]
            sample_points2 = keypoints2[samples[:, 1]][:, :2]
            H = find_homography(sample_points1, sample_points2)
            if np.allclose(np.linalg.det(H), 0):
                # H is degenerate, skip it
                continue

            # Calculate reprojection error and see which matches agree with H (inliers)
            error = calculate_reprojection_error(H, matched_keypoints1, matched_keypoints2)
            inlier_indices = np.transpose(np.nonzero(error < reprojection_threshold)).flatten()
            if best_inlier_indices is None or inlier_indices.shape[0] > best_inlier_indices.shape[0]:
                best_inlier_indices = inlier_indices
                best_H = H

        return best_inlier_indices, best_H

    def classify_tile(self, tile):
        templates = []
        tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

        max_matches = 0
        winning_template = None
        _, tile_des = self.sift.detectAndCompute(tile_gray, None)
        if tile_des is None:
            return None

        for letter, template, template_des in self.template_descriptors:
            matches = self.match_keypoints(tile_des, template_des, threshold=0.8)
            if len(matches) == 0:
                continue
            inlier_indices, _ = self.refine_match(tile_des, template_des, matches, reprojection_threshold=10)
            if inlier_indices is None:
                continue
            if max_matches < len(inlier_indices):
                max_matches = len(inlier_indices)
                winning_template = template
            # bf = cv2.BFMatcher()
            # matches = bf.match(des_tile, des_template)
            #img = cv2.drawMatches(tile_gray, kp_tile, template, kp_template, matches[inlier_indices], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        print('max matches:', max_matches)
        if max_matches > 0:
            show_images([tile_gray, winning_template])
        else:
            show_images([tile_gray])
        return None

    def process(self, inputs, visualize=False):
        tiles = inputs['tiles']
        labels = [self.classify_tile(t) for t in tiles]

        print('labels:', labels)
        outputs = {'tile_labels': labels}
        if visualize:
            img_copy = np.copy(inputs['img'])
            outputs['debug_img'] = img_copy

        return outputs
