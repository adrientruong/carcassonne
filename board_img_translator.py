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
            new_board_origin_x = int(round_nearest(board_origin[0], 64) / 64)
            new_board_origin_y = int(round_nearest(board_origin[1], 64) / 64)

            self.origin = np.array([new_board_origin_x, new_board_origin_y])

    def new_translator_with_img_origin(self, origin):
        return BoardImageTranslator(self.H, self.H_inv, origin)

    def get_img_origin(self):
        return self.board_p_to_img_p((0, 0))

    def board_p_to_img_p(self, p):
        p = np.array(p)
        p += self.origin
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

        return board_p

    def board_pos_to_tile_bbox(self, p):
        tl = self.board_p_to_img_p(p)
        br = self.board_p_to_img_p((p[0] + 64, p[1] + 64))

        if tl[0] > br[0]:
            temp_tl = tl
            tl = br
            br = temp_tl

        return (tl, br)

    def tile_img_at_pos(self, pos):
        x_index, y_index = pos
        tl = self.board_p_to_img_p(x_index * 64, y_index * 64)
        tr = self.board_p_to_img_p(x_index * 64 + 64, y_index * 64)
        bl = self.board_p_to_img_p(x_index * 64, y_index * 64 + 64)
        br = self.board_p_to_img_p(x_index * 64 + 64, y_index * 64 + 64)
        min_x = int(min(tl[0], bl[0]))
        min_y = int(min(tl[1], tr[1]))
        max_x = int(max(tr[0], br[0]))
        max_y = int(max(bl[1], br[1]))

        if min_x < -50:
            return None
        if min_y < -50:
            return None
        if max_x >= img.shape[1] + 50:
            return None
        if max_y >= img.shape[0] + 50:
            return None

        min_x = max(0, min_x)
        min_y = max(0, min_y)

        world_points = np.array([[0, 0],
                                [64, 0],
                                [64, 64],
                                [0, 64]], dtype=np.float32)
        img_points = np.array([tl, tr, br, bl])
        transform = cv2.getPerspectiveTransform(img_points, world_points)
        tile = cv2.warpPerspective(img, transform, (64, 64))

        return tile