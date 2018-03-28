import numpy as np
import cv2

from collections import defaultdict
from pipeline import *
from canny import *
from hough import *
from blur import *
from find_intersections import *
from find_homography import *
from dilate import *
from resize import *
from tile_classifier import *
from find_tiles import *
from board_reconstructor import *
from piece_detector import *
from game import *
from default_tiles import get_default_tiles
from better_tile_classifier import TileClassifierNew

from utils import *
from game_state_drawer import *
from translator_drawer import *
from timeit import default_timer as timer

class Tracker():
    def __init__(self):
        CANNY_WEAK_THRESHOLD = 100
        CANNY_STRONG_THRESHOLD = 200
        HOUGH_RHO_STEP = 1
        HOUGH_THETA_STEP = np.pi/180
        HOUGH_THRESHOLD = 50
        HOUGH_MIN_LINE_LEN = 30
        HOUGH_MAX_LINE_GAP = 30

        full_pipeline = Pipeline()
        full_pipeline.add_step(Resize(max_width=800))
        full_pipeline.add_step(Blur(9, 0.75))
        full_pipeline.add_step(CannyEdgeDetector(CANNY_WEAK_THRESHOLD, CANNY_STRONG_THRESHOLD))
        full_pipeline.add_step(Dilate('edges', 2))
        full_pipeline.add_step(HoughLineProbabilisticTransform(HOUGH_RHO_STEP, HOUGH_THETA_STEP, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP))
        full_pipeline.add_step(FindIntersections())
        full_pipeline.add_step(RANSACHomography())
        full_pipeline.add_step(FindTilesNew(padding=0))
        full_pipeline.add_step(TileClassifier())
        full_pipeline.add_step(PieceDetector())
        full_pipeline.add_step(ReconstructBoard())

        partial_pipeline = Pipeline()
        partial_pipeline.add_step(Resize(max_width=800))
        partial_pipeline.add_step(Blur(9, 0.75))
        partial_pipeline.add_step(FindTilesNew(padding=0))
        partial_pipeline.add_step(TileClassifier())
        partial_pipeline.add_step(PieceDetector())
        partial_pipeline.add_step(ReconstructBoard())

        self.full = full_pipeline
        self.partial = partial_pipeline
        self.tile_classifier = TileClassifierNew()

        self.prev_outputs = None
        self.last_full_run = 0

        self.last_frame = None

        player1 = Player('Adrien')
        player2 = Player('Kyle')
        self.previous_game_state = None
        self.game_state = CarcGameState([player1, player2])

        self.translator = None
        self.board_homography = None
        self.last_turn = None

    def diff_between_frames(self, frame1, frame2):
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame1, frame2).sum()

        return diff

    def heat_map_between_frames(self, frame1, frame2):
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame1, frame2)

        return diff

    def classify_tile_at_pos(self, img, pos):
        known_features = self.game_state.required_features_for_placement(pos)
        tile = self.tile_classifier.classify(img, known_features)

        return tile

    def add_turn_for_frame(self, turn, frame):
        self.previous_game_state = self.game_state
        self.game_state = self.game_state.after_playing_turn(turn)
        self.last_turn = turn
        self.last_frame = frame

        for player, points in zip(self.game_state.players, self.game_state.player_points):
            print('{}: {}'.format(player.name, points))
        print('-' * 80)
        self.show_game_state()

    def add_piece_to_last_turn_with_frame(self, frame):
        self.game_state = self.previous_game_state
        new_turn = Turn(self.last_turn.tile, self.last_turn.position, place_piece=True)
        self.game_state = self.game_state.after_playing_turn(new_turn)
        self.last_turn = new_turn
        self.last_frame = frame

        for player, points in zip(self.game_state.players, self.game_state.player_points):
            print('{}: {}'.format(player.name, points))
        print('-' * 80)
        self.show_game_state()

    def show_game_state(self):
        board_img = draw_game_state(self.game_state)
        cv2.imshow('board state', board_img)

    def process_frame(self, frame):
        #print('last full run:', self.last_full_run)
        # don't process if last frame is similar to last processed_frame
        if self.last_frame is None:
            self.last_frame = frame
            return

        if self.game_state.turn_num == 0:
            outputs = self.full.run({'img': frame}, visualize=False)
            if outputs is None:
                return

            default_tiles = get_default_tiles()
            classified_tile = outputs['classified_tiles'][0]
            location = outputs['classified_locations'][0]
            self.translator = outputs['board_img_translator']
            img_origin = self.translator.board_p_to_img_p(location)
            self.translator = self.translator.new_translator_with_img_origin(img_origin)
            game_tile = default_tiles[classified_tile.letter].tile_by_rotating(classified_tile.rotation)
            #game_tile = default_tiles['B'].tile_by_rotating(classified_tile.rotation)
            #show_image(game_tile.img, 'detected img')
            turn = Turn(game_tile, (0, 0))
            self.add_turn_for_frame(turn, frame)
            self.board_homography = outputs['board_homography']
            cv2.imshow('homography', self.board_homography)
        else:
            cv2.imshow('homography', self.board_homography)

            heat_map = self.heat_map_between_frames(self.last_frame, frame)
            resizer = Resize(max_width=800)
            #last_frame = resizer.process({'img': self.last_frame})['img']
            #frame_smaller = resizer.process({'img': frame})['img']
            heat_map = resizer.process({'img': heat_map})['img']

            pos_with_new_tile = None
            #translator = translator.new_translator_with_img_origin(self.reference_board_origin)
            NEW_TILE_DIFF_THRESHOLD = 75000
            max_diff = 0
            raw_img = None
            positions_to_check = list(self.game_state.placeable_positions())
            positions_to_check.append(self.last_turn.position)
            for pos in positions_to_check:
                #print('checking pos:', pos)
                #cv2.imshow('heat map', heat_map)
                tl, br = self.translator.board_pos_to_tile_bbox(pos)
                width = br[0] - tl[0]
                height = br[1] - tl[1]
                img_area = heat_map[tl[1]:tl[1]+height, tl[0]:tl[0]+width]
                diff = img_area.sum()
                print('heatmap diff:', diff)
                if diff > NEW_TILE_DIFF_THRESHOLD and diff > max_diff:
                    max_diff = diff
                    pos_with_new_tile = pos
                    raw_img = img_area

            if pos_with_new_tile is not None:
                print('max_diff:', max_diff)
                if pos_with_new_tile == self.last_turn.position:
                    self.add_piece_to_last_turn_with_frame(frame)
                    return

                thresholded = np.copy(heat_map)
                thresholded[thresholded > 50] = 255
                thresholded[thresholded <= 50] = 0
                kernel = np.ones((5,5),np.uint8)
                thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

                resizer = Resize(max_width=800)
                thresholded = resizer.process({'img': thresholded})['img']

                #show_image(thresholded, 'thresh!')
                im2, contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    print('Found difference with 0 contours. Aborting!')
                    return

                biggest_contour = sorted(contours, key=cv2.contourArea)[-1]
                epsilon = cv2.arcLength(biggest_contour, True) * 0.1
                box = cv2.approxPolyDP(biggest_contour, epsilon, True)

                #rect = cv2.minAreaRect(biggest_contour)
                #box = cv2.boxPoints(rect)
                #box = approx
                #box = np.int0(box)
                box = np.squeeze(box)

                cv2.drawContours(thresholded,[box],0,(100),5)
                #print('rect:', rect)

                #show_image(thresholded, 'heat_map')
                #cv2.imshow('heat_map', thresholded)
                print('pos with new tile:', pos_with_new_tile)
                print('max diff', max_diff)


                resized_frame = resizer.process({'img': frame})['img']

                if box.shape[0] == 4:
                    print('refining!')
                    box_board_p = []
                    for img_p in box:
                        board_p = self.translator.img_p_to_board_p(img_p)
                        box_board_p.append(board_p)
                    self.translator.refine(box, box_board_p, resized_frame)

                board_homography = np.copy(resized_frame)
                draw_translator(board_homography, self.translator)
                self.board_homography = board_homography

                new_tile_img = self.translator.tile_img_at_pos(resized_frame, pos_with_new_tile)
                #new_tile_img = self.translator.tile_img_from_box(resized_frame, box)
                game_tile = self.classify_tile_at_pos(new_tile_img, pos_with_new_tile)
                #cv2.imshow('new_tile_img', new_tile_img)
                #show_image(new_tile_img)
                turn = Turn(game_tile, pos_with_new_tile)
                self.add_turn_for_frame(turn, frame)
                #show_image(new_tile_img)
                #show_images([raw_img, new_tile_img])
                #show_image(new_tile_img, 'new tile')
        self.show_game_state()
