import numpy as np
import cv2
from utils import round_nearest

class BoardImageTranslator():
    def __init__(self):
        self.origin = np.array([0, 0])

    def fit(self, img_points, board_points, img):
        result = self.calculate_homography(img_points, board_points, img)
        if not result:
            return False

        H, H_inv = result
        self.H = H
        self.H_inv = H_inv
        self.img_points = img_points
        self.board_points = board_points

        return True

    def refine(self, new_img_points, new_board_points, img):
        new_img_points = np.array(new_img_points)
        new_board_points = np.array(new_board_points) * 64

        self.img_points = np.append(self.img_points, new_img_points, axis=0)
        self.board_points = np.append(self.board_points, new_board_points, axis=0)

        H, H_inv = self.calculate_homography(self.img_points, self.board_points, img)
        self.H = H
        self.H_inv = H_inv

    def is_reasonable_homography(self, H_inv):
        def board_p_to_image_p(p):
            p = np.array(p)
            p = np.append(p, 1)
            image_point = H_inv.dot(p.reshape(3, 1))
            image_point = np.squeeze(image_point)
            image_point = image_point[:2] / image_point[2]
            image_point = image_point.astype('int64')
            return image_point

        for x_i, y_i in np.ndindex((5, 5)):
            x = x_i * 64
            y = y_i * 64
            p1 = board_p_to_image_p((x, y))
            p2 = board_p_to_image_p((x, y + 64))
            p3 = board_p_to_image_p((x + 64, y))
            dist_p1_p2 = np.linalg.norm(p1 - p2)
            dist_p1_p3 = np.linalg.norm(p1 - p3)
            if max(dist_p1_p2, dist_p1_p3) > 80 and np.abs(dist_p1_p3 - dist_p1_p2) >= 0.5 * min(dist_p1_p2, dist_p1_p3):
                print('distances:', dist_p1_p2, dist_p1_p3)
                return False

        return True

    def calculate_homography(self, img_points, board_points, img):
        # make copy b/c of weird layout bug
        # open cv complains if we use image_points directly
        copy = []
        for p in img_points:
            copy.append(p)
        copy = np.array(copy, dtype=np.float32)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        cv2.cornerSubPix(gray_img, copy, (5, 5), (-1, -1), criteria)
        img_points = copy

        H_inv, _ = cv2.findHomography(board_points, img_points, cv2.RANSAC, 5.0)
        if H_inv is None:
            print('Could not generate homography. Aborting.')
            return None

        if not self.is_reasonable_homography(H_inv):
            print('Generated homography does not look correct. Aborting.')
            return None

        H, _ = cv2.findHomography(img_points, board_points, cv2.RANSAC, 5.0)

        return H, H_inv

    def new_translator_with_img_origin(self, origin):
        new = BoardImageTranslator()
        new.img_points = self.img_points
        new.board_points = self.board_points
        new.H = self.H
        new.H_inv = self.H_inv

        board_origin = self.img_p_to_board_p(origin)
        new.origin = np.array(board_origin)

        return new

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

        nearest_x = int(round_nearest(board_p[0], 64) / 64)
        nearest_y = int(round_nearest(board_p[1], 64) / 64)

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
        for img_p in box:
            board_p = self.img_p_to_board_p(img_p)
            point_map[board_p] = img_p
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