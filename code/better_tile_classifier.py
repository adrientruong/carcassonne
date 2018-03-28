import cv2

from default_tiles import *
from utils import *
from canny import *

class TileClassifierNew():
    def __init__(self):
        self.default_tiles = get_default_tiles()

    def normalized_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        mean = img.mean()
        std = img.std()
        std = max(std, 0.0000001)
        img -= mean
        img /= std

        return img

    def normalized_mask(self, mask):
        mask = np.copy(mask)
        mask = mask.astype(np.int32)
        mask[mask == 0] = -1
        mask[mask > 0] = 1

        return mask

    def masked_img(self, img):
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

        canny = CannyEdgeDetector(100, 175)
        mask = canny.process({'img': img})['edges']
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def cross_correlation(self, img1, img2):
        return (img1 * img2).sum()

    def classify(self, unknown_tile_img, known_features={}):
        normalized_unknown = self.normalized_img(unknown_tile_img)
        unknown_mask = self.masked_img(unknown_tile_img)
        normalized_unknown_mask = self.normalized_mask(unknown_mask)


        best_score = 0
        best_tile = None
        scores_by_letter = {}
        for letter, tile in self.default_tiles.items():
            best_tile_score = 0
            for r in range(4):
                rotated_tile = tile.tile_by_rotating(r)
                if not rotated_tile.has_features(known_features):
                    continue

                tile_img = rotated_tile.img[4:-4, 4:-4]
                resized_tile_img = cv2.resize(tile_img, (64, 64))
                normalized_tile_img = self.normalized_img(resized_tile_img)
                tile_mask = self.masked_img(resized_tile_img)
                normalized_tile_mask = self.normalized_mask(tile_mask)
                img_score = self.cross_correlation(normalized_tile_img, normalized_unknown)
                # img_score /= 10000
                # mask_score = self.cross_correlation(normalized_tile_mask, normalized_unknown_mask)
                # mask_score /= 10000
                # score = 50 * img_score + 50 * mask_score
                score = img_score
                if score > best_score:
                    best_score = score
                    best_tile = rotated_tile
                best_tile_score = max(best_tile_score, score)
            if best_tile_score > 0:
                scores_by_letter[letter] = best_tile_score

        #for key in sorted(scores_by_letter.keys(), key=lambda k: scores_by_letter[k]):
        #    print(key, scores_by_letter[key])

        return best_tile
