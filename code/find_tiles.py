from pipeline import PipelineStep
import cv2
import numpy as np
from utils import *
from itertools import combinations, product
from canny import *

class FindTilesNew(PipelineStep):
    def __init__(self, padding):
        self.padding = padding

    def process(self, inputs, visualize=False):
        H = inputs['H']
        img = inputs['img']
        tiles = []
        rects = []
        locations = []
        H_inv = inputs['H_inv']
        def board_p_to_image_p(x, y):
            img_p = H_inv.dot(np.array([x, y, 1]))
            img_p = img_p[:2] / img_p[2]
            img_p = img_p.astype(np.float32)
            return img_p

        #cv2.imshow('homography', inputs['debug_img'])
        for y_index, x_index in product(range(-10, 10), range(-10, 10)):
            padding = self.padding

            tl = board_p_to_image_p(x_index * 64 - padding, y_index * 64 - padding)
            tr = board_p_to_image_p(x_index * 64 + 64 + padding, y_index * 64 - padding)
            bl = board_p_to_image_p(x_index * 64 - padding, y_index * 64 + 64 + padding)
            br = board_p_to_image_p(x_index * 64 + 64 + padding, y_index * 64 + 64 + padding)
            min_x = int(min(tl[0], bl[0]))
            min_y = int(min(tl[1], tr[1]))
            max_x = int(max(tr[0], br[0]))
            max_y = int(max(bl[1], br[1]))

            if min_x < -50:
                continue
            if min_y < -50:
                continue
            if max_x >= img.shape[1] + 50:
                continue
            if max_y >= img.shape[0] + 50:
                continue

            min_x = max(0, min_x)
            min_y = max(0, min_y)

            world_points = np.array([[0, 0],
                                    [64, 0],
                                    [64, 64],
                                    [0, 64]], dtype=np.float32)
            img_points = np.array([tl, tr, br, bl])
            transform = cv2.getPerspectiveTransform(img_points, world_points)
            tile = cv2.warpPerspective(img, transform, (64, 64))
            #show_image(tile)
            rects.append(((min_x, min_y), (max_x, max_y)))
            tiles.append(tile)
            locations.append((x_index, y_index))

        outputs = {'tiles': tiles, 'tile_locations': locations}

        if visualize:
            debug_img = np.copy(img)
            for p1, p2 in rects:
                cv2.rectangle(debug_img, p1, p2, (0, 0, 255))
            outputs['debug_img'] = debug_img

        return outputs


class FindTiles(PipelineStep):
    def __init__(self, padding):
        self.padding = padding

    def process(self, inputs, visualize=False):
        H = inputs['H']
        img = inputs['img']
        (min_x, max_x), (min_y, max_y) = get_xy_range_of_transformation(img, H)
        t_x = -np.floor(min_x)
        t_y = -np.floor(min_y)
        t = np.array([[1, 0, t_x],
                      [0, 1, t_y],
                      [0, 0, 1]])
        new_H = t.dot(H)
        new_w = int(np.ceil(max_x - min_x))
        new_h = int(np.ceil(max_y - min_y))
        new_w = min(2000, new_w)
        new_h = min(2000, new_h)
        birds_eye_img = cv2.warpPerspective(img, new_H, (new_w, new_h))

        tiles = []
        rects = []
        first_x = int(t_x) % 64
        first_y = int(t_y) % 64
        locations = []
        for y, x in product(range(first_y, birds_eye_img.shape[0], 64), range(first_x, birds_eye_img.shape[1], 64)):
            y_index = int((y - first_y) / 64)
            x_index = int((x - first_x) / 64)

            padding = self.padding
            if x-padding < 0:
                continue
            if y-padding < 0:
                continue
            if x + 64 + padding >= birds_eye_img.shape[1]:
                continue
            if y + 64 + padding >= birds_eye_img.shape[0]:
                continue

            #show_image(tile)

            rects.append(((x-padding, y-padding), (x+64+padding, y+64+padding)))
            tile = birds_eye_img[y-padding:y+64+padding, x-padding:x+64+padding]
            tiles.append(tile)
            locations.append((x_index, y_index))

        outputs = {'tiles': tiles, 'tile_locations': locations}

        if visualize:
            debug_img = np.copy(img)
            for p1, p2 in rects:
                cv2.rectangle(debug_img, p1, p2, (0, 0, 255))
            outputs['debug_img'] = debug_img

        return outputs
