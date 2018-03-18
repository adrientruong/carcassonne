import numpy as np
import cv2

def round_nearest(x, base=5):
    return int(base * round(float(x)/base))

def show_image(img, name='image'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_images(images):
    new_h = 0
    new_w = 0
    for img in images:
        new_h = max(new_h, img.shape[0])
        new_w += img.shape[1]

 
    new_img = np.zeros((new_h, new_w, 3), dtype=images[0].dtype)
    current_x = 0
    for i, img in enumerate(images):
        if len(img.shape) < len(new_img.shape):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        new_img[:img.shape[0], current_x:current_x+img.shape[1]] = img
        current_x += img.shape[1]
    cv2.imshow('merged', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_xy_range_of_transformation(img, H):
    min_x = 100000
    max_x = -10000
    min_y = 100000
    max_y = -100000
    for y, x in np.ndindex(img.shape[:2]):
        new_p = H.dot(np.array([x, y, 1]))
        new_p = new_p[:2] / new_p[2]
        min_x = min(min_x, new_p[0])
        max_x = max(max_x, new_p[0])
        min_y = min(min_y, new_p[1])
        max_y = max(max_y, new_p[1])

    return (min_x, max_x), (min_y, max_y)

def warp_image(img, H):
    (min_x, max_x), (min_y, max_y) = get_xy_range_of_transformation(img, H)
    t_x = -np.floor(min_x)
    t_y = -np.floor(min_y)
    t = np.array([[1, 0, t_x],
                  [0, 1, t_y],
                  [0, 0, 1]])
    new_H = t.dot(H)
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))
    warped_img = cv2.warpPerspective(img, new_H, (new_w, new_h))

    return warped_img


def intersection_of_polar_lines(img, l1, l2):
    rho1, theta1 = l1.flatten()
    rho2, theta2 = l2.flatten()
    cos1, sin1 = np.cos(theta1), np.sin(theta1)
    cos2, sin2 = np.cos(theta2), np.sin(theta2)

    # Ap = b
    A = np.array([[cos1, sin1],
                  [cos2, sin2]])
    b = np.array([rho1, rho2])
    p = np.linalg.inv(A).dot(b)

    p = p.astype('int64')
    if p[0] < 0 or p[0] >= img.shape[1]:
        return None
    if p[1] < 0 or p[1] >= img.shape[0]:
        return None

    return p

def find_intersection(img, l1, l2):
    point = np.cross(l1, l2)
    if point[2] == 0:
        return None

    point = point[:2] / point[2]
    point = point.astype('int64')
    if point[0] < 0 or point[0] >= img.shape[1]:
        return None
    if point[1] < 0 or point[1] >= img.shape[0]:
        return None

    return point

def point_on_la_line_at_x(line, x):
    m = -line[0] / line[1]
    b = -line[2] / line[1]
    y = m * x + b
    return np.array([x, y])    

def polar_line_from_point_theta(p, theta):
    d = np.linalg.norm(p)
    d_theta = np.arctan2(p[1], p[0])
    rho = d * np.cos(d_theta - theta)

    if theta < 0:
        theta += np.deg2rad(180)
        rho = -rho
    if theta >= np.deg2rad(135):
        theta -= np.deg2rad(180)
        rho = -rho

    return np.array([rho, theta])

def la_line_from_point_slope(p1, m):
    p2 = np.array([p1[0] + 100, p1[1] + (m * 100)])
    return la_line_from_segment(np.concatenate([p1, p2]))

def la_line_from_segment(segment):
    point1 = np.append(segment[:2], 1)
    point2 = np.append(segment[2:], 1)

    return np.cross(point1, point2)

def polar_line_from_segment(segment):
    p1 = segment[:2]
    p2 = segment[2:]
    d = np.linalg.norm(p2 - p1)
    theta = -np.arctan2(p2[0] - p1[0], p2[1] - p1[1])
    return polar_line_from_point_theta(p1, theta)
    #rho = np.abs(p2[0] * p1[1] - p2[1] * p1[0]) / np.linalg.norm(p2 - p1)
    #theta = -np.arctan2(p2[0] - p1[0], p2[1] - p1[1])

    # if theta < 0:
    #     rho = -rho

    # return np.array([rho, theta])

def draw_2point_lines(img, lines, color=(0, 255, 0)):
    polar_lines = []
    for line in lines:
        polar = polar_line_from_segment(line)
        polar_lines.append(polar)
    polar_lines = np.array(polar_lines)
    draw_polar_lines(img, polar_lines, color=color)

def draw_2point_line_segments(img, lines, color=(0, 255, 0)):
    for x1,y1,x2,y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=2)

def draw_la_lines(img, lines, color=(0, 255, 0)):
    for line in lines:
        m = -line[0] / line[1]
        b = -line[2] / line[1]

        x1 = 0
        y1 = int(m * x1 + b)
        x2 = img.shape[1]
        y2 = int(m * x2 + b)

        cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=2)

def draw_polar_lines(img, lines, color=(0, 255, 0)):
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

        cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=2)