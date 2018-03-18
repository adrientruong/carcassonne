import numpy as np
import cv2
from utils import round_nearest

class BoardImageTranslator():
    def __init__(self, H, H_inv, img_origin=None):
        self.H = H
        self.H_inv = H_inv

        self.origin = np.array([0, 0])
        if img_origin is not None:
            board_origin = self.img_p_to_board_p(img_origin)
            self.origin = np.array(board_origin)

    def new_translator_with_img_origin(self, origin):
        return BoardImageTranslator(self.H, self.H_inv, origin)

    def get_img_origin(self):
        return self.board_p_to_img_p((0, 0))

    def board_p_to_img_p(self, p):
        p = np.array(p)
        p += self.origin
        p *= 64
        p = np.append(p, 1)
        image_point = self.H_inv.dot(p.reshape(3, 1))
        image_point = np.squeeze(image_point)
        image_point = image_point[:2] / image_point[2]
        image_point = image_point.astype(np.int32)

        return image_point

    def img_p_to_board_p(self, p):
        p = np.array(p)
        p = np.append(p, 1)
        board_p = self.H.dot(p.reshape(3, 1))
        board_p = np.squeeze(board_p)
        board_p = board_p[:2] / board_p[2]
        board_p = board_p.astype(np.int32)
        board_p -= self.origin

        print('actual board p', board_p, 'from ', p)

        nearest_x = int(round_nearest(board_p[0], 64) / 64)
        nearest_y = int(round_nearest(board_p[1], 64) / 64)

        print('nearestxy:', (nearest_x, nearest_y))

        return (nearest_x, nearest_y)

    def board_pos_from_box(self, box):
        min_x = 1000
        min_y = 1000
        for img_p in box:
            board_p = self.img_p_to_board_p(img_p)
            min_x = min(board_p[0], min_x)
            min_y = min(board_p[1], min_y)

        return (min_x, min_y)

    def tile_img_from_box(self, img, box):
        point_map = {}
        print('box:', box)
        for img_p in box:
            board_p = self.img_p_to_board_p(img_p)
            print('board_p', board_p)
            point_map[board_p] = img_p
            print('setting point map!')
        print(sorted(point_map.keys()))
        img_pts = [point_map[key] for key in sorted(point_map.keys())]
        tl = img_pts[0]
        bl = img_pts[1]
        tr = img_pts[2]
        br = img_pts[3]

        world_points = np.array([[0, 0],
                                [64, 0],
                                [64, 64],
                                [0, 64]], dtype=np.float32)
        img_points = np.array([tl, tr, br, bl], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(img_points, world_points)
        tile = cv2.warpPerspective(img, transform, (64, 64))

        return tile

    def board_pos_to_tile_bbox(self, pos):
        x, y = pos
        tl = self.board_p_to_img_p((x, y))
        tr = self.board_p_to_img_p((x + 1, y))
        bl = self.board_p_to_img_p((x, y + 1))
        br = self.board_p_to_img_p((x + 1, y + 1))

        img_x, img_y, w, h = cv2.boundingRect(np.array([tl, br, bl, br]))

        return ((img_x, img_y), (img_x+w, img_y+h))

    def tile_img_at_pos(self, img, pos):
        x_index, y_index = pos
        tl = self.board_p_to_img_p((x_index, y_index))
        tr = self.board_p_to_img_p((x_index + 1, y_index))
        br = self.board_p_to_img_p((x_index + 1, y_index + 1))
        bl = self.board_p_to_img_p((x_index, y_index + 1))

        world_points = np.array([[0, 0],
                                [64, 0],
                                [64, 64],
                                [0, 64]], dtype=np.float32)
        img_points = np.array([tl, tr, br, bl], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(img_points, world_points)
        tile = cv2.warpPerspective(img, transform, (64, 64))

        return tile