from pipeline import PipelineStep
import cv2
import numpy as np
from utils import *
from itertools import product
from sklearn.cluster import MeanShift, KMeans
from scipy import stats

import matplotlib.pyplot as plt

class FindIntersections(PipelineStep):
    def label_vert_horiz_polar_lines(self, polar_lines):
        thetas = polar_lines[:, 1]
        degrees = np.rad2deg(thetas)
        kmeans = KMeans(n_clusters=2).fit(degrees.reshape(-1, 1))
        labels = kmeans.labels_
        # ensure vert lines become 0
        # vert lines have 0 degrees
        centers = np.squeeze(kmeans.cluster_centers_)
        if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
            adjusted_labels = np.copy(labels)
            adjusted_labels[labels == 0] = 1
            adjusted_labels[labels == 1] = 0
            labels = adjusted_labels
        return labels

    def group_polar_lines(self, img, polar_lines, line_segments):
        thetas = polar_lines[:, 1]
        perp_theta = thetas.mean() + np.deg2rad(90)
        #print('thetas:', thetas)
        #print('perp_theta:', perp_theta)
        #print('d:', discriminator)

        center = np.array([img.shape[1], img.shape[0]]) / 2
        perp_line = polar_line_from_point_theta(center, perp_theta)
        points = [intersection_of_polar_lines(img, l, perp_line) for l in polar_lines]
        for i in range(len(points)):
            if points[i] is None:
                points[i] = [0, 0]
        points = np.array(points)

        img = np.copy(img)
        draw_polar_lines(img, np.array([perp_line]))
        for p, pl, ls in zip(points, polar_lines, line_segments):
            draw_2point_line_segments(img, np.array([ls]))
            if p is None:
                #draw_polar_lines(img, np.array([pl]))
                draw_2point_line_segments(img, np.array([ls]))
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ms = MeanShift(bandwidth=20, cluster_all=False)
        labels = ms.fit_predict(points)
        labels[labels == -1] = np.max(labels) + 1

        label_mapping = {}
        next_label = 0
        ordering = None
        if np.std(points[:, 0]) > np.std(points[:, 1]):
            ordering = points[:, 0]
        else:
            ordering = points[:, 1]
        for i in np.argsort(ordering):
            label = labels[i]
            if label not in label_mapping:
                label_mapping[label] = next_label
                next_label += 1
            labels[i] = label_mapping[label]

        # for i in range(np.max(labels) + 1):
        #     print('i:', i, 'dis: ', rhos[labels == i])

        # plt.plot(np.sort(discriminator))
        # plt.ylabel('discriminator')
        # plt.show()

        return labels, perp_line

    def filter_outliers(self, polar_lines, line_segments):
        best_polar_lines = None
        best_line_segments = None
        best_theta = None
        N = 50
        thetas = polar_lines[:, 1]
        thetas = np.rad2deg(thetas)
        for i in range(N):
            sample_index = np.random.randint(0, polar_lines.shape[0])
            sample_theta = thetas[sample_index]
            inlier_indices = np.nonzero(np.abs(thetas - sample_theta) < 10)[0]
            #print(inlier_indices)
            if best_polar_lines is None or inlier_indices.shape[0] > best_polar_lines.shape[0]:
                best_polar_lines = polar_lines[inlier_indices]
                best_line_segments = line_segments[inlier_indices]
                best_theta = sample_theta
                #print('found new best:', best_polar_lines.shape[0])

        #print('best theta:', best_theta)
        return best_polar_lines, best_line_segments

    def process(self, inputs, visualize=False):
        line_segments = inputs['line_segments']

        if line_segments is None or line_segments.shape[0] <= 1:
            print('Not enough lines to find intersection!')
            return None

        polar_lines = np.array([polar_line_from_segment(s) for s in line_segments])
        vert_horiz_labels = self.label_vert_horiz_polar_lines(polar_lines)
        indices1 = np.nonzero(vert_horiz_labels == 0)
        indices2 = np.nonzero(vert_horiz_labels == 1)
        line_segments1 = line_segments[indices1]
        line_segments2 = line_segments[indices2]
        polar_lines1 = polar_lines[indices1]
        polar_lines2 = polar_lines[indices2]

        polar_lines1, line_segments1 = self.filter_outliers(polar_lines1, line_segments1)
        polar_lines2, line_segments2 = self.filter_outliers(polar_lines2, line_segments2)

        img = inputs['img']
        labels1, perp_line1 = self.group_polar_lines(img, polar_lines1, line_segments1)
        labels2, perp_line2 = self.group_polar_lines(img, polar_lines2, line_segments2)

        intersection_labels = {}
        intersection_labels_contrib = {}
        intersection_bins = np.zeros(inputs['img'].shape[:2])
        for (l1, l1_s, i), (l2, l2_s, j) in product(zip(polar_lines1, line_segments1, labels1), zip(polar_lines2, line_segments2, labels2)):
            point = intersection_of_polar_lines(img, l1, l2)
            if point is None:
                continue

            l1_len = np.linalg.norm(l1_s[:2] - l1_s[2:])
            l2_len = np.linalg.norm(l2_s[:2] - l2_s[2:])
            dist_l1 = min(np.linalg.norm(l1_s[:2] - point), np.linalg.norm(l1_s[2:] - point))
            #dist_l1 *= 0.3
            dist_l1 = max(10, dist_l1)
            dist_l2 = min(np.linalg.norm(l2_s[:2] - point), np.linalg.norm(l2_s[2:] - point))
            #dist_l2 *= 0.3
            dist_l2 = max(10, dist_l1)

            dist_l1 = 10
            dist_l2 = 10

            vote = (1/dist_l1 * l1_len) + (1/dist_l2 * l2_len)
            intersection_bins[point[1], point[0]] += vote

            key = (point[0], point[1])
            if key not in intersection_labels_contrib or vote > intersection_labels_contrib[key]:
                intersection_labels[key] = (i, j)
                intersection_labels_contrib[key] = vote

        for (y, x), count in np.ndenumerate(intersection_bins):
            if count == 0:
                continue

            window_size = 25
            x_range = (max(0, x - window_size), min(x + window_size, intersection_bins.shape[1] - 1))
            y_range = (max(0, y - window_size), min(y + window_size, intersection_bins.shape[0] - 1))
            window = intersection_bins[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            if np.max(window) > count:
               intersection_bins[y, x] = 0

        intersection_img = np.zeros(intersection_bins.shape)
        intersection_img[intersection_bins > 0] = 255
        intersection_img = intersection_img.astype('uint8')
        filtered_intersections = np.transpose(np.nonzero(intersection_bins > 0))
        filtered_intersections = np.flip(filtered_intersections, axis=1)

        outputs = {'intersections': filtered_intersections, 'intersection_labels': intersection_labels, 'intersections_img': intersection_img}
        if visualize:
            img_copy = np.copy(inputs['img'])
            colors = [
                (255, 97, 0),
                (0, 0, 255),
                (255, 0, 246),
                (255, 195, 0),
                (165, 255, 0),
                (0, 255, 38),
                (0, 255, 255),
                (0, 161, 255),
                (0, 0, 0),
                (255, 255, 255),
            ]

            #draw_polar_lines(img_copy, np.array([perp_line1]), color=(255, 255, 255))
            #draw_polar_lines(img_copy, np.array([perp_line2]), color=(255, 255, 255))
            for i in range(line_segments1.shape[0]):
                ls = line_segments1[i]
                pl = polar_lines1[i]
                label = labels1[i]
                color = colors[label % len(colors)]
                x1, y1, x2, y2 = ls.flatten()
                draw_2point_line_segments(img_copy, [ls], color=color)
                center = (int((x2 + x1)/2) + 10, int((y2 + y1)/2) + 10)
                #cv2.putText(img_copy, str(label), org=center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))
            for i in range(line_segments2.shape[0]):
                ls = line_segments2[i]
                pl = polar_lines2[i]
                label = labels2[i]
                color = colors[label % len(colors)]
                x1, y1, x2, y2 = ls.flatten()
                center = (int((x2 + x1)/2) + 10, int((y2 + y1)/2) + 10)
                draw_2point_line_segments(img_copy, [ls], color=color)
                #cv2.putText(img_copy, str(int(label)), org=center, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))

            #draw_2point_line_segments(img_copy, line_segments2)
            #for p in intersections:
                #cv2.circle(img_copy, center=tuple(p), radius=2, thickness=1, color=(255, 0, 0))

            # strong_intersections = np.argsort(intersection_bins, axis=None)
            # for i in strong_intersections[-30:]:
            #     p = np.unravel_index(i, intersection_bins.shape)
            #     cv2.circle(img_copy, center=(p[1], p[0]), radius=2, thickness=2, color=(0, 0, 255))

            for p, count in np.ndenumerate(intersection_bins):
                if count > 0:
                    red = np.log(count) / np.log(np.max(intersection_bins)) * 255
                    cv2.circle(img_copy, center=(p[1], p[0]), radius=2, thickness=2, color=(0, 0, red))
                    i, j = intersection_labels[(p[1], p[0])]
                    cv2.putText(img_copy, '({}, {})'.format(i, j), org=(p[1], p[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))

            outputs['debug_img'] = img_copy

        return outputs
