from pipeline import PipelineStep
import cv2
import numpy as np

class PieceDetector(PipelineStep):
    def process(self, inputs, visualize=False):
        img = inputs['img']
        img_hsv =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        RED_RANGE = (np.array([0, 125, 125]), np.array([7, 255, 255]))
        BLUE_RANGE = (np.array([90, 125, 125]), np.array([120, 255, 255]))
        YELLOW_RANGE = (np.array([27, 125, 125]), np.array([32, 255, 255]))
        BLACK_RANGE = (np.array([0, 0, 0]), np.array([255, 100, 100]))
        #GREEN_RANGE = (np.array([30, 230, 230]), np.array([70, 255, 255]))
        colors = [BLUE_RANGE, RED_RANGE, YELLOW_RANGE, BLACK_RANGE]
        found_pieces = []
        for lower, upper in colors:
            img_copy = np.copy(img)
            mask = cv2.inRange(img_hsv, lower, upper)
            result = cv2.bitwise_and(img_copy, img_copy, mask=mask)
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = result.copy()
            binary[result > 0] = 255

            #kernel = np.ones((3, 3), np.uint8)
            #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # cv2.imshow('threshold', mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            good_contours = []
            for c in contours:
                area = cv2.contourArea(c)
                if area > 200 and area < 1000 and len(c) < 150:
                    print('area:', area)
                    print('len of countour:', len(c))
                    good_contours.append(c)

            print('found pieces:', len(good_contours))

            found_pieces.append(good_contours)

        outputs = {'pieces': found_pieces}

        if visualize:
            debug_img = np.copy(img)
            BLUE = (255, 0, 0)
            RED = (0, 0, 255)
            YELLOW = (0, 255, 255)
            BLACK = (0, 0, 0)
            # intentional mix order so we can see outline
            COLORS = [RED, YELLOW, BLACK, BLUE]
            for pieces, color in zip(found_pieces, COLORS):
                for c in pieces:
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(img,(x,y),(x+w,y+h), color,2)
                    #cv2.drawContours(debug_img, pieces, -1, color, 2)
            outputs['debug_img'] = debug_img

        return outputs