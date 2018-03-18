from pipeline import PipelineStep
import cv2
import numpy as np
from canny import *
import string
from utils import *
from collections import defaultdict
from kmeans import *

class ClassifiedTile():
    def __init__(self, raw_img, matched_tile, letter, rotation):
        self.raw_img = raw_img
        self.matched_tile = matched_tile
        self.letter = letter
        self.rotation = rotation

class FeaturedBasedTileClassifier(PipelineStep):
    def __init__(self):
        self.edge_detector = CannyEdgeDetector(30, 100)
        self.templates = self.generate_templates()

    def generate_templates(self):
        templates = []
        for letter in string.ascii_uppercase[:24]:
            name = 'data/tiles/' + letter + '.png'
            tile_img = cv2.imread(name)
            tile_img = cv2.resize(tile_img, (64, 64))
            t = self.generate_template(tile_img)
            #cv2.imwrite('data/tiles/templates/' + letter + '.png', t[0])
            templates.append(t)
        return templates

    def generate_template(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        left_edge = img_hsv[:, :20]
        right_edge = img_hsv[:, -20:]
        top_edge = img_hsv[:20, :]
        bottom_edge = img_hsv[-20:, :]
        center = img_hsv[20:-20, 20:-20, :]

        left_avg_hue = left_edge[:, :, 0].mean()
        right_avg_hue = right_edge[:, :, 0].mean()
        top_avg_hue = top_edge[:, :, 0].mean()
        bottom_avg_hue = bottom_edge[:, :, 0].mean()
        center_avg_hue = center[:, :, 0].mean()

        features = np.array([left_avg_hue, right_avg_hue, top_avg_hue, bottom_avg_hue, center_avg_hue])

        return features, img

    def classify_tile(self, raw_tile):
        winning_scores = []
        winning_rotations = []
        for template_features, raw_template in self.templates:
            max_score = 100000
            winning_rotation = -1
            for r in range(4):
                rotated_tile = np.rot90(raw_tile, k=r)
                tile_features, _ = self.generate_template(rotated_tile)
                score = np.linalg.norm(tile_features - template_features)
                if score < max_score:
                    max_score = score
                    winning_rotation = 4-r
                #show_images([tile, rotated_template])

            winning_scores.append(max_score)
            winning_rotations.append(winning_rotation)

        winning_scores = np.array(winning_scores)
        #print(np.sort(winning_scores).astype(np.int32))

        label = np.argmax(winning_scores)
        rotation = winning_rotations[label]
        letter = string.ascii_uppercase[label]

        # winning_template_edges = self.templates[label][0]
        # winning_template_edges = np.rot90(winning_template_edges, k=rotation)
        winning_template = self.templates[label][1]
        winning_template = np.rot90(winning_template, k=rotation)
        # ratio = np.mean(winning_template_edges) / np.mean(tile)
        # if ratio > 1.5:
        #     return None

        #show_images([tile, winning_template_edges])
        show_images([cv2.cvtColor(raw_tile, cv2.COLOR_BGR2GRAY), cv2.cvtColor(winning_template, cv2.COLOR_BGR2GRAY)])

        classified = ClassifiedTile(raw_tile, winning_template, letter, rotation)

        return classified

    def process(self, inputs, visualize=False):
        tiles = inputs['tiles']
        classified_tiles = [self.classify_tile(t) for t in tiles]

        locations = inputs['tile_locations']
        actual_classsified_tiles = []
        actual_locations = []
        for tile, location in zip(classified_tiles, locations):
            if tile is not None:
                actual_classsified_tiles.append(tile)
                actual_locations.append(location)


        outputs = {'classified_tiles': actual_classsified_tiles, 'classified_locations': actual_locations}
        if visualize:
            img_copy = np.copy(inputs['img'])
            outputs['debug_img'] = img_copy

        return outputs

class TileClassifier(PipelineStep):
    def __init__(self):
        self.edge_detector = CannyEdgeDetector(100, 175)
        self.templates = self.generate_templates()

    def normalized_img(self, img):
        img = np.copy(img)
        img = img.astype(np.int32)
        img[img == 0] = -1
        img[img == 255] = 1
        # img = img.astype('float32')
        # mean = img.mean()
        # std = img.std()
        # std = max(std, 0.0000001)
        # img -= mean
        # img /= std

        return img

    def normalized_cross_correlation(self, raw_tile, template):
        normalized_tile = self.normalized_img(raw_tile)
        normalized_template = self.normalized_img(template)

        max_score = -1.0
        diff_y = normalized_tile.shape[0] - normalized_template.shape[0]
        diff_x = normalized_tile.shape[1] - normalized_template.shape[1]
        H, W = normalized_template.shape

        for y, x in np.ndindex((diff_y + 1, diff_x + 1)):
            window = normalized_tile[y:y+H, x:x+W]
            #print('sum:', (window * normalized_template).sum())
            max_score = max(max_score, (window * normalized_template).sum())

        #res = cv2.matchTemplate(raw_tile, template, cv2.TM_CCORR)
        #min_val, max_score, min_loc, max_loc = cv2.minMaxLoc(res)
        #print('max score:', max_score)

        return max_score

    def better_score(self, raw_tile, template):
        difference = cv2.bitwise_xor(raw_tile, template)
        addition = cv2.bitwise_or(raw_tile, template)
        show_images([raw_tile, template, difference, addition])

        return 0

    def generate_template(self, img):
        #kmeans = BoardKMeansIdentifier(K=3)
        #template = kmeans.process({'img': img}, visualize=True)['debug_img']
        #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #template = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #template = template[:, :, 0]

        # img = cv2.blur(img, (2, 2))
        # template = self.edge_detector.process({'img': img})['edges']
        # kernel = np.ones((2, 2), np.uint8)
        # template = cv2.dilate(template, kernel, iterations=1)

        green_lower = np.array([35, 50, 125])
        green_upper = np.array([75, 255, 255])
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
        mask = cv2.bitwise_not(green_mask)

        result = cv2.bitwise_and(img, img, mask=mask)

        #mask = mask[5:-5, 5:-5]
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            #print('area:', cv2.contourArea(c))
            if cv2.contourArea(c) < 30:
                x, y, w, h = cv2.boundingRect(c)
                cv2.drawContours(mask, [c], -1, (0, 0, 255), cv2.FILLED)
                mask[y:y+h, x:x+w] = 0

        # kernel = np.ones((4, 4), np.uint8)
        # mask = cv2.dilate(mask, kernel, iterations=1)

        #show_images([result, mask, img])
        #print("-------")

        #template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return mask

    def generate_templates(self):
        templates = []
        for letter in string.ascii_uppercase[:24]:
            name = 'data/tiles/' + letter + '.png'
            tile_img = cv2.imread(name)
            tile_img = tile_img[4:-4, 4:-4]
            tile_img = cv2.resize(tile_img, (64, 64))
            t = self.generate_template(tile_img)
            # if letter == 'V':
            #     copy = np.copy(t)
            #     for y, x in np.ndindex(t.shape):
            #         if y + 4 < t.shape[0]:
            #             copy[y, x] = t[y + 4, x]
            #     t = copy

            cv2.imwrite('data/tiles/templates/' + letter + '.png', t)
            templates.append((t, tile_img))
        return templates

    def generate_features(self, img):
        left_indicator = False
        right_indicator = False
        chopped_img = img[10:-10, :]
        for y in range(chopped_img.shape[0]):
            if y - 1 < 0 or y + 1 >= chopped_img.shape[0]:
                continue
            left_line = chopped_img[y-1:y+1, :10]
            right_line = chopped_img[y-1:y+1, -10:]
            if np.all(left_line > 0):
                left_indicator = True
            if np.all(right_line > 0):
                right_indicator = True

        top_indicator = False
        bottom_indicator = False
        chopped_img = img[:, 10:-10]
        for x in range(chopped_img.shape[1]):
            if x - 1 < 0 or x + 1 >= chopped_img.shape[1]:
                continue
            top_line = chopped_img[:10, x-1:x+1]
            bottom_line = chopped_img[-10:, x-1:x+1]
            if np.all(top_line > 0):
                top_indicator = True
            if np.all(bottom_line > 0):
                bottom_indicator = True

        # def percentage_filled(img):
        #     total = img.shape[0] * img.shape[1]
        #     filled = np.count_nonzero(img)
        #     print('total:', total)
        #     print('filled:', filled)
        #     print('percentage:', filled/total)
        #     return filled / total
        # DENSE_THRESHOLD = 0.5
        # left_dense_indicator = percentage_filled(img[10:-10, :20]) > DENSE_THRESHOLD
        # right_dense_indicator = percentage_filled(img[10:-10, -20:]) > DENSE_THRESHOLD
        # top_dense_indicator = percentage_filled(img[:20, 10:-10]) > DENSE_THRESHOLD
        # bottom_dense_indicator = percentage_filled(img[-20:, 10:-10]) > DENSE_THRESHOLD


        return np.array([left_indicator, top_indicator, right_indicator, bottom_indicator])


    def classify_tile(self, raw_tile):
        center = raw_tile[5:-5, 5:-5]
        tile_edges = self.edge_detector.process({'img': center})['edges']
        #show_images([raw_tile, tile_edges])
        if np.mean(tile_edges) < 15:
            return None

        tile = self.generate_template(raw_tile)

        #show_images([raw_tile, tile])

        tile_feature = self.generate_features(tile)

        #print('tile_angle:', tile_angle)
        winning_scores = []
        winning_rotations = []
        winning_features = []
        i = 0
        #print('tile_feature:', tile_feature)
        #show_images([raw_tile, tile])
        for template, raw_template in self.templates:
            max_score = -1
            winning_rotation = -1
            winning_feature = None
            #print('letter:', string.ascii_uppercase[i])

            for r in range(4):
                rotated_template = np.rot90(template, k=r)
                template_feature = self.generate_features(rotated_template)
                #print('tile_feature:', tile_feature)
                #print('template feature:', template_feature)
                #show_images([raw_tile, tile, rotated_template])
                if not np.array_equal(tile_feature, template_feature):
                    pass
                    #continue

                score = self.normalized_cross_correlation(tile, rotated_template)
                if score > max_score:
                    #show_images([raw_tile, tile, rotated_template])
                    max_score = score
                    winning_rotation = r
                    winning_feature = template_feature
                #show_images([tile, rotated_template])

            winning_scores.append(max_score)
            winning_rotations.append(winning_rotation)
            winning_features.append(winning_feature)
            i += 1

        winning_scores = np.array(winning_scores)

        #print(np.sort(winning_scores).astype(np.int32))
        has_V_or_U_in_top5 = False
        for i in np.argsort(-winning_scores)[:5]:
            letter = string.ascii_uppercase[i]
            if letter == 'U' or letter == 'V':
                has_V_or_U_in_top5 = True
        #if has_V_or_U_in_top5:
        #    V_rotated = 

        label = np.argmax(winning_scores)
        rotation = winning_rotations[label]
        letter = string.ascii_uppercase[label]

        winning_template = self.templates[label][1]
        winning_template = np.rot90(winning_template, k=rotation)

        sorted_scores = np.sort(winning_scores)
        delta = sorted_scores[-1] - sorted_scores[-2]
        if delta < 200 and False:
            winning_feature = winning_features[label]
            print('tile_feature:', tile_feature)
            print('template feature:', winning_feature)

            letters = string.ascii_uppercase[:24]
            for i in np.argsort(winning_scores):
                print(letters[i], winning_scores[i])
            print('-' * 80)
            show_images([raw_tile, tile, self.templates[label][0], winning_template])

        # show_images([tile, winning_template_edges])
        #show_image(raw_tile)
        # show_images([cv2.cvtColor(raw_tile, cv2.COLOR_BGR2GRAY), cv2.cvtColor(winning_template, cv2.COLOR_BGR2GRAY)])
        # show_image(raw_tile)
        # show_image(tile)
        # show_image(self.templates[label][0])
        classified = ClassifiedTile(raw_tile, winning_template, letter, rotation)

        return classified

    def process(self, inputs, visualize=False):
        tiles = inputs['tiles']
        classified_tiles = [self.classify_tile(t) for t in tiles]

        locations = inputs['tile_locations']
        actual_classsified_tiles = []
        actual_locations = []
        for tile, location in zip(classified_tiles, locations):
            if tile is not None:
                actual_classsified_tiles.append(tile)
                actual_locations.append(location)

        outputs = {'classified_tiles': actual_classsified_tiles, 'classified_locations': actual_locations}

        if visualize:
            img_copy = np.copy(inputs['img'])
            outputs['debug_img'] = img_copy

        return outputs
