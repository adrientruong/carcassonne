from pipeline import PipelineStep
import cv2
import numpy as np
from utils import *

class CorrectPerspective(PipelineStep):
    def process(self, inputs, visualize=False):
        img = inputs['img']
        H = inputs['H']
        (min_x, max_x), (min_y, max_y) = get_xy_range_of_transformation(img, H)
        t_x = -np.floor(min_x)
        t_y = -np.floor(min_y)
        t = np.array([[1, 0, t_x],
                      [0, 1, t_y],
                      [0, 0, 1]])
        new_H = t.dot(H)
        new_w = int(np.ceil(max_x - min_x))
        new_h = int(np.ceil(max_y - min_y))
        birds_eye_img = cv2.warpPerspective(img, new_H, (new_w, new_h))

        outputs = {'img': birds_eye_img, 'debug_img': birds_eye_img}

        return outputs

# class CorrectPerspective(PipelineStep):
#   def process(self, inputs, visualize=False):
#       img = inputs['img']
#       camera_points = inputs['camera_points']
#       side = 64*5
#       world_points = np.array([[0, 0],
#                                [side, 0],
#                                [side, side],
#                                [0, side]], dtype='float32')
#       for i in range(world_points.shape[0]):
#           world_points[i] += np.array([200, 250])

#       M = cv2.getPerspectiveTransform(camera_points, world_points)
#       top_down = cv2.warpPerspective(img, M, (1000, 1500))

#       outputs = {'img': top_down, 'debug_img': top_down}
#       return outputs

#   def process(self, inputs, visualize=False):
#       img = inputs['img']
#       line_points1 = inputs['points1']
#       line_points2 = inputs['points2']

#       vanishing_point1 = compute_vanishing_point(line_points1)
#       vanishing_point2 = compute_vanishing_point(line_points2)

#       horizon_line = np.cross(vanishing_point1, vanishing_point2)


#       outputs = {'img': top_down, 'debug_img': top_down}
#       return outputs

# def compute_rectified_image(im, H):
#     new_x = np.zeros(im.shape[:2])
#     new_y = np.zeros(im.shape[:2])
#     for y in range(im.shape[0]): # height
#         for x in range(im.shape[1]): # width
#             new_location = H.dot([x, y, 1])
#             new_location /= new_location[2]
#             new_x[y,x] = new_location[0]
#             new_y[y,x] = new_location[1]
#     offsets = (new_x.min(), new_y.min())
#     new_x -= offsets[0]
#     new_y -= offsets[1]
#     new_dims = (int(np.ceil(new_y.max()))+1,int(np.ceil(new_x.max()))+1)

#     H_inv = np.linalg.inv(H)
#     new_image = np.zeros(new_dims)

#     for y in range(new_dims[0]):
#         for x in range(new_dims[1]):
#             old_location = H_inv.dot([x+offsets[0], y+offsets[1], 1])
#             old_location /= old_location[2]
#             old_x = int(old_location[0])
#             old_y = int(old_location[1])
#             if old_x >= 0 and old_x < im.shape[1] and old_y >= 0 and old_y < im.shape[0]:
#                 new_image[y,x] = im[old_y, old_x]

#     return new_image, offsets

# def compute_vanishing_point(points):
#     def line_from_points(points):       
#         point1 = np.append(points[0], 1)
#         point2 = np.append(points[1], 1)

#         return np.cross(point1, point2)

#     l1 = line_from_points(points[:2, :])
#     l2 = line_from_points(points[2:, :])
#     intersection = np.cross(l1, l2)
#     intersection = intersection[:2] / intersection[2]

#     return intersection