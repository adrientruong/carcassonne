from pipeline import PipelineStep
import cv2
import numpy as np
from itertools import product
from utils import *
from board_img_translator import BoardImageTranslator

class RANSACHomography(PipelineStep):
    def convert_to_homo(self, points):
        return np.hstack([points, np.ones((points.shape[0], 1))])

    def find_homography(self, points1, points2):
        points1 = self.convert_to_homo(points1)
        points2 = self.convert_to_homo(points2)

        #Hp1 = p2
        #p1^T H^T = p2^T
        return np.linalg.lstsq(points1.T, points2.T)[0].T

    def calculate_reprojection_error(self, H, points1, points2):
        points1_homo = self.convert_to_homo(points1)
        projected_points = H.dot(points1_homo.T).T
        projected_points = projected_points[:, :2] / projected_points[:, 2].reshape((projected_points.shape[0], 1))
        error = np.linalg.norm(points2 - projected_points, axis=1)
        return error

    def process(self, inputs, visualize=False):
        image_points = inputs['intersections']
        labels = inputs['intersection_labels']

        if image_points.shape[0] == 0:
            print('No suitable image points found!')

            return None

        N = 1000
        best_H = None
        best_inlier_indices = None

        board_points = []
        for p in image_points:
            coord = labels[tuple(p.flatten())]
            x_coord = coord[0]
            y_coord = coord[1]
            #if x_coord == 5:
                #x_coord -= 1
            board_points.append([x_coord * 64, y_coord * 64])
        board_points = np.array(board_points)

        board_img_translator = BoardImageTranslator()
        result = board_img_translator.fit(image_points, board_points, inputs['img'])
        if not result:
            return None

        outputs = {
            'H': board_img_translator.H,
            'H_inv': board_img_translator.H_inv,
            'board_img_translator': board_img_translator
        }
        visualize = True
        if visualize:
            debug_img = np.copy(inputs['img'])
            for x, y in product(range(-10, 10), range(-10, 10)):
                image_point = board_img_translator.board_p_to_img_p((x, y))
                cv2.circle(debug_img, center=tuple(image_point), radius=2, thickness=2, color=(0, 0, 255))
                cv2.putText(debug_img, '({}, {})'.format(x, y), org=tuple(image_point), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))
            outputs['debug_img'] = debug_img
            outputs['board_homography'] = debug_img

        return outputs



'''
COMPUTE_RECTIFIED_IMAGE
Arguments:
    im - an image
    H - a homography matrix that rectifies the image
Returns:
    new_image - a new image matrix after applying the homography
    offset - the offest in the image.
'''
def compute_rectified_image(im, H):
    new_x = np.zeros(im.shape[:2])
    new_y = np.zeros(im.shape[:2])
    for y in range(im.shape[0]): # height
        for x in range(im.shape[1]): # width
            new_location = H.dot([x, y, 1])
            new_location /= new_location[2]
            new_x[y,x] = new_location[0]
            new_y[y,x] = new_location[1]
    offsets = (new_x.min(), new_y.min())
    new_x -= offsets[0]
    new_y -= offsets[1]
    new_dims = (int(np.ceil(new_y.max()))+1,int(np.ceil(new_x.max()))+1)

    H_inv = np.linalg.inv(H)
    new_image = np.zeros(new_dims)

    for y in range(new_dims[0]):
        for x in range(new_dims[1]):
            old_location = H_inv.dot([x+offsets[0], y+offsets[1], 1])
            old_location /= old_location[2]
            old_x = int(old_location[0])
            old_y = int(old_location[1])
            if old_x >= 0 and old_x < im.shape[1] and old_y >= 0 and old_y < im.shape[0]:
                new_image[y,x] = im[old_y, old_x]

    return new_image, offsets

