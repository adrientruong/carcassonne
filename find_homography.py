from pipeline import PipelineStep
import cv2
import numpy as np
from itertools import product
from utils import *

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

        # make copy b/c of weird layout bug
        # open cv complains if we use image_points directly
        copy = []
        for p in image_points:
            copy.append(p)
        copy = np.array(copy, dtype=np.float32)
        gray_img = cv2.cvtColor(inputs['img'], cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        cv2.cornerSubPix(gray_img, copy, (5, 5), (-1, -1), criteria)
        image_points = copy

        H, _ = cv2.findHomography(image_points, board_points, cv2.RANSAC, 5.0)
        if H is None:
            print('Coud not generate homography!')
            return None

        H_inv, _ = cv2.findHomography(board_points, image_points, cv2.RANSAC, 5.0)
        outputs = {'H': H, 'H_inv': H_inv}
        if visualize:
            debug_img = np.copy(inputs['img'])
            for x, y in product(range(-10, 10), range(-10, 10)):
                point = np.array([x * 64, y * 64, 1])
                image_point = H_inv.dot(point.reshape(3, 1))
                image_point = np.squeeze(image_point)
                image_point = image_point[:2] / image_point[2]
                image_point = image_point.astype('int64')
                cv2.circle(debug_img, center=tuple(image_point), radius=2, thickness=2, color=(0, 0, 255))
                cv2.putText(debug_img, '({}, {})'.format(x, y), org=tuple(image_point), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))
            outputs['debug_img'] = debug_img

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

