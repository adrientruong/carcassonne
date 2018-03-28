from pipeline import PipelineStep
import cv2
import numpy as np
from utils import *
from itertools import combinations, product

class FindTiles(PipelineStep):
    def fit(self, img, lines1, lines2, l1, l2):
        p1 = find_intersection(img, l1, l2)
        if p1 is None:
            return None

        max_area = None
        best_p2 = None
        best_p3 = None
        best_p4 = None
        for l3, l4 in product(lines1, lines2):
            if np.all(np.equal(l2, l4)) or np.all(np.equal(l1, l3)):
                continue

            p2 = find_intersection(img, l1, l4)
            p3 = find_intersection(img, l3, l4)
            p4 = find_intersection(img, l2, l3)
            if p2 is None or p3 is None or p4 is None:
                continue

            side1 = np.linalg.norm(p2 - p1)
            side2 = np.linalg.norm(p3 - p2)
            side3 = np.linalg.norm(p4 - p3)
            side4 = np.linalg.norm(p4 - p1)

            if np.abs(side3 - side1) > 10:
                continue
            if np.abs(side4 - side2) > 10:
                continue
            if np.abs(side4 - side1) > 5:
                continue

            #print(side1, side2, side3, side4)

            area = side1 * side2
            if (max_area is None or area > max_area) and area > (20 * 20) and area < (60 * 60):
                max_area = area
                best_p2 = p2
                best_p3 = p3
                best_p4 = p4

        if best_p2 is None:
            return None

        return np.array([p1, best_p2, best_p3, best_p4])

    def process(self, inputs, visualize=False):
        line_segments = inputs['line_segments']
        lines = np.array([la_line_from_segment(s) for s in line_segments])
        slopes = np.array([(y2 - y1) / (x2-x1) for x1, y1, x2, y2 in line_segments])
        indices1 = np.transpose(np.nonzero(slopes<0)).flatten()
        indices2 = np.transpose(np.nonzero(slopes>=0)).flatten()
        lines1 = lines[indices1]
        lines2 = lines[indices2]
        intersections = []
        img = inputs['img']

        tiles = []
        print('num lines1:', lines1.shape[0])
        print('num lines2:', lines2.shape[0])
        for l1, l2 in product(lines1, lines2):
            best_tile = self.fit(img, lines1, lines2, l1, l2)
            if best_tile is not None:
                tiles.append(best_tile)
            print('found tile ', len(tiles))
            if len(tiles) == 1:
                break

        tiles = np.array(tiles)
        print(tiles)

        outputs = {'tiles': tiles}

        if visualize:
            debug_img = np.copy(inputs['img'])
            lines = []
            for tile in tiles:
                p1, p2, p3, p4 = tile
                l1 = np.concatenate([p1, p2])
                l2 = np.concatenate([p2, p3])
                l3 = np.concatenate([p3, p4])
                l4 = np.concatenate([p4, p1])
                lines.extend([l1, l2, l3, l4])

            lines = np.array(lines)
            draw_2point_line_segments(debug_img, lines)

            for tile in tiles:
                p1, p2, p3, p4 = tile
                cv2.circle(debug_img, center=tuple(p1), radius=2, thickness=3, color=(100, 0, 100))
                cv2.circle(debug_img, center=tuple(p2), radius=2, thickness=3, color=(255, 150, 0))
                cv2.circle(debug_img, center=tuple(p3), radius=2, thickness=3, color=(0, 0, 0))
                cv2.circle(debug_img, center=tuple(p4), radius=2, thickness=3, color=(135, 50, 0))
            outputs['debug_img'] = debug_img

        return outputs