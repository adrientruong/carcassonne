from pipeline import PipelineStep
import cv2
import numpy as np
from utils import *
from collections import defaultdict

class ComputeK(PipelineStep):
    def process(self, inputs, visualize=False):
        img = inputs['img']
        lines = inputs['grid_lines']
        lines_by_angle = defaultdict(list)
        for line in lines:
            _, theta = line.flatten()
            theta = np.rad2deg(theta)
            theta = round_nearest(theta, 10)
            if theta < 0:
                theta += 180
            lines_by_angle[theta].append(line)

        def generate_2_points(line):
            rho, theta = line.flatten()
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            return np.array([[x1, y1],
                             [x2, y2]])

        vanishing_points = []
        theta = 90
        line1 = lines_by_angle[theta][1]
        line2 = lines_by_angle[theta][2]
        line3 = lines_by_angle[theta][3]
        line4 = lines_by_angle[theta][4]
        points1 = generate_2_points(line1)
        points2 = generate_2_points(line2)
        points3 = generate_2_points(line3)
        points4 = generate_2_points(line4)
        points_set_1 = np.concatenate([points1, points2], axis=0)
        points_set_2 = np.concatenate([points3, points4], axis=0)
        vp_1 = compute_vanishing_point(points_set_1)
        vp_2 = compute_vanishing_point(points_set_2)
        print('vps')
        print(vp_1)
        print(vp_2)
        l = np.cross(vp_1, vp_2)
        print('vanishing line:')
        print(l)
        H_1 = np.array([[1, 0, 0],
                        [0, 1, 0],
                        l])
        rectified = compute_rectified_image(img, H_1)

        outputs = {'img': rectified, 'debug_img': rectified}
        return outputs

def compute_vanishing_point(points):
    def line_from_points(points):       
        point1 = np.append(points[0], 1)
        point2 = np.append(points[1], 1)

        return np.cross(point1, point2)

    l1 = line_from_points(points[:2, :])
    l2 = line_from_points(points[2:, :])
    intersection = np.cross(l1, l2)
    intersection = intersection[:2] / intersection[2]

    return intersection

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
