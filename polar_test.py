import cv2
from utils import *

img = cv2.imread('data/boards/board.jpg')
line = [20, np.deg2rad(90)]

center = np.array([img.shape[1], img.shape[0]]) / 2
theta2 = np.arctan(center[1] / center[0])
perp_theta = line[1] + np.deg2rad(90)
perp_r = np.linalg.norm(center) * np.cos(theta2 - perp_theta)
perp_line = np.array([perp_r, perp_theta])

perp_line = polar_line_from_point_theta(center, perp_theta)

print('line:', line)
print('perp line:', perp_line)
lines = [line]

#draw_polar_lines(img, np.array(lines))

line_segment = np.array([0, 300, 150, 300])
polar_line = polar_line_from_segment(line_segment)
#draw_2point_line_segments(img, np.array([line_segment]))
draw_polar_lines(img, np.array([polar_line]))

cv2.imshow('lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()