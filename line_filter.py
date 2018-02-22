from pipeline import PipelineStep
import numpy as np
from collections import defaultdict
from utils import *

class LineFilter(PipelineStep): 
    def process(self, inputs, visualize=False):
        lines = inputs['lines']

        #round_degrees(lines)
        convert_to_degrees(lines)
        normalize_lines(lines)
        #lines = lines[:3]
        lines = filter_similar_lines(lines)

        lines_histo = group_lines_by_deg(lines)
        dominant_theta_count = 0
        dominant_theta = 0
        for theta in lines_histo:
            count = len(lines_histo[theta])
            print((theta, count))
            if count > dominant_theta_count:
                dominant_theta = theta
                dominant_theta_count = count

        dominant_theta = np.abs(dominant_theta)
        actual_dominant_theta = np.mean(lines_histo[dominant_theta], axis=0)[1]
        dominant_perpendicular_theta_lowest_diff = 1000
        dominant_perpendicular_theta = 0
        for line in lines:
            _, theta = line
            diff = np.abs(np.abs(dominant_theta - theta) - 90)
            if diff < dominant_perpendicular_theta_lowest_diff:
                dominant_perpendicular_theta = theta
                dominant_perpendicular_theta_lowest_diff = diff
                if diff <= 5:
                    break
        if dominant_perpendicular_theta < 0:
            dominant_perpendicular_theta += 180
        print('remaining lines')
        print(lines)
        final_lines = []

        for line in lines:
            _, theta = line
            if theta < 0:
                theta += 180
            if np.abs((theta - dominant_theta)) <= 5 or np.abs(theta - dominant_perpendicular_theta) <= 5:
                #if theta == 102:
                #    continue
                final_lines.append(line)

        #final_lines = filter_offspaced_lines(final_lines)
        final_lines = np.array(final_lines)

        print('dominant_theta:', dominant_theta)
        print('matching theta:', dominant_perpendicular_theta)
        print('dominant_theta count:', dominant_theta)

        #lines = list(filter(is_horiz_or_vert, lines))
        #final_lines = lines

        # lines_by_deg = group_lines_by_deg(lines)
        # final_lines = []
        # for _, lines_wih_deg in lines_by_deg.items():
        #     filtered_lines = filter_offspaced_lines(lines_wih_deg)
        #     final_lines.extend(filtered_lines)

        print('final lines:', np.array(final_lines))

        # for line in final_lines:
        #    line[1] += dominant_theta

        # for line in final_lines:
        #     line[0] = 124

        for line in final_lines:
            line[1] = np.deg2rad(line[1])

        outputs = {'img': inputs['img'], 'grid_lines': final_lines}

        if visualize:
            img_copy = np.copy(inputs['img'])
            draw_polar_lines(img_copy, final_lines)
            outputs['debug_img'] = img_copy

        return outputs

def is_horiz_or_vert(line):
    theta = line[1]
    theta = np.abs(theta)
    return np.abs(theta - 90) <= 20 or np.abs(theta) <= 20

def normalize_lines(lines):
    for line in lines:
        if line[0] < 0:
            line[0] *= -1
            line[1] -= 180

def round_degrees(lines):
    for line in lines:
        theta_rad = line[1]
        theta_deg = np.rad2deg(theta_rad)
        line[1] = round_nearest(theta_deg, 5)

def convert_to_degrees(lines):
    for line in lines:
        theta_rad = line[1]
        line[1] = round_nearest(np.rad2deg(theta_rad), 1)

def filter_similar_lines(lines):
    seen_lines = defaultdict(bool)
    deduped_lines = []
    for line in lines:
        rho, theta = line.flatten()
        rounded_rho = round_nearest(rho, 20)

        #print((rho, theta), ":", (rounded_rho, rounded_theta))
        if seen_lines[(rho, theta)]:
            continue

        RHO_TOLERANCE = 40
        THETA_TOLERANCE = 5
        for r_tol in range(-RHO_TOLERANCE, RHO_TOLERANCE + 1):
            for t_tol in range(-THETA_TOLERANCE, THETA_TOLERANCE + 1):
                seen_lines[(rho + r_tol, theta + t_tol)] = True
        deduped_lines.append(line)

    return np.array(deduped_lines)

def filter_offspaced_lines(lines):
    # sort by rho
    lines = sorted(lines, key=lambda line: line[0])
    lines_by_rho_diff = defaultdict(list)
    last_rho = lines[0][0]
    most_common_diff = 0
    most_common_diff_count = 0
    for line in lines[1:]:
        rho, _ = line
        diff = rho - last_rho
        last_rho = rho

        rounded_diff = round_nearest(diff, 10)
        lines_with_rho_diff = lines_by_rho_diff[rounded_diff]
        lines_with_rho_diff.append(line)

        if len(lines_with_rho_diff) > most_common_diff_count:
            most_common_diff = rounded_diff
            most_common_diff_count = len(lines_with_rho_diff)

    filtered_lines = []
    correct_rho = round_nearest(lines_by_rho_diff[most_common_diff][0][0], 5)
    for line in lines:
        rho, _ = line
        rounded_rho = round_nearest(rho, 5)
        diff = np.abs(correct_rho - rounded_rho)
        multiplier =  round_nearest(diff / most_common_diff, 1)
        if np.abs((most_common_diff * multiplier) - diff) <= (10 * multiplier):
            filtered_lines.append(line)

    return filtered_lines

def group_lines_by_deg(lines):
    lines_by_deg = defaultdict(list)
    for line in lines:
        _, theta = line
        theta = round_nearest(theta, 5)
        lines_by_deg[theta].append(line)
    return lines_by_deg