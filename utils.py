import numpy as np
import cv2

def round_nearest(x, base=5):
    return int(base * round(float(x)/base))

def two_points_to_polar(line):
    p1 = line[:2]
    p2 = line[2:]
    rho = np.abs(p2[0] * p1[1] - p2[1] * p1[0]) / np.linalg.norm(p2 - p1)
    theta = -np.arctan2(p2[0] - p1[0], p2[1] - p1[1])

    if theta < 0:
        rho = -rho

    return np.array([rho, theta])

def draw_2point_lines(img, lines):
    polar_lines = []
    for line in lines:
        polar = two_points_to_polar(line)
        polar_lines.append(polar)
    polar_lines = np.array(polar_lines)
    draw_polar_lines(img, polar_lines)

def draw_2point_line_segments(img, lines):
    for x1,y1,x2,y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def draw_polar_lines(img, lines):
    if len(lines.shape) == 0:
        return

    for line in lines:
        rho, theta = line.flatten()

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2)