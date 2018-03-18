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

        self.prev_outputs = None
        self.last_full_run = 0

        self.last_frame = None

        player1 = Player('Adrien')
        player2 = Player('Kyle')
        self.game_state = CarcGameState([player1, player2])

        self.reference_board_origin = None

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

    def normalized_translator(self, outputs):
        board_p_to_img_p = outputs['board_p_to_img_p']
        if self.reference_board_origin is None:
            return

        img_p_to_board_p = outputs['img_p_to_board_p']
        board_origin = img_p_to_board_p(self.reference_board_origin)
        new_board_origin_x = int(round_nearest(board_origin[0], 64) / 64)
        new_board_origin_y = int(round_nearest(board_origin[1], 64) / 64)

        def normalized_img_p_to_board_p(p):
            board_p = img_p_to_board_p(p)
            return (board_p[0] - new_board_origin_x, board_p[1] - new_board_origin_y)

        def normalized_board_p_to_img_p(p):
            p = (p[0] + new_board_origin_x, p[1] + new_board_origin_y)
            return board_p_to_img_p(p)

        board_pos_to_tile_bbox = outputs['board_pos_to_tile_bbox']
        def normalized_board_pos_to_tile_bbox(p):
            p = (p[0] + new_board_origin_x, p[1] + new_board_origin_y)
            return board_pos_to_tile_bbox(p)

        outputs['img_p_to_board_p'] = normalized_img_p_to_board_p
        outputs['board_p_to_img_p'] = normalized_board_p_to_img_p
        outputs['board_pos_to_tile_bbox'] = normalized_board_pos_to_tile_bbox

    def process_frame(self, frame):
        #print('last full run:', self.last_full_run)
        # don't process if last frame is similar to last processed_frame
        if self.last_frame is None:
            self.last_frame = frame
            return

        heat_map = self.heat_map_between_frames(self.last_frame, frame)
        # if heat_map.sum() < 3000000:
        #     # not different enough
        #     return
        resizer = Resize(max_width=800)
        last_frame = resizer.process({'img': self.last_frame})['img']
        frame_smaller = resizer.process({'img': frame})['img']
        heat_map = resizer.process({'img': heat_map})['img']

        outputs = self.full.run({'img': frame}, visualize=False)
        if outputs is None:
            return

        possible_new_tile_pos = self.game_state.placeable_positions()
        translator = outputs['board_img_translator']
        translator = translator.new_translator_with_img_origin(self.reference_board_origin)

        if self.game_state.turn_num == 0:
            default_tiles = get_default_tiles()
            print(outputs['classified_tiles'])
            classified_tile = outputs['classified_tiles'][0]
            self.reference_board_origin = translator.get_img_origin()
            print('img_origin:', self.reference_board_origin)
            game_tile = default_tiles[classified_tile.letter].tile_by_rotating(classified_tile.rotation)
            turn = Turn(game_tile, (0, 0))
            self.game_state = self.game_state.after_playing_turn(turn)
            cv2.imshow('homography', outputs['board_homography'])
            return

        pos_with_new_tile = None
        max_diff = 0
        for pos in self.game_state.placeable_positions():
            tl, br = translator.board_pos_to_tile_bbox(pos)
            width = br[0] - tl[0]
            height = br[1] - tl[1]
            img_area = heat_map[tl[1]:tl[1]+height, tl[0]:tl[0]+width]
            diff = img_area.sum()
            print('heatmap diff:', diff)
            if diff > max_diff:
                max_diff = diff
                pos_with_new_tile = pos
        NEW_TILE_DIFF_THRESHOLD = 50000
        if max_diff > NEW_TILE_DIFF_THRESHOLD:
            print('pos with new tile:', pos_with_new_tile)
            print('max diff', max_diff)

            self.last_frame = frame
        #show_images([last_frame, frame_smaller, heat_map])

        return

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame1, frame2).sum()

        if timer() - self.last_full_run > 30:
            print('Starting from scracth!')
            self.prev_outputs = None

        if self.prev_outputs is None:
            outputs = self.full.run({'img': frame}, visualize=False)
            self.last_full_run = timer()
        else:
            self.prev_outputs['img'] = frame
            outputs = self.partial.run(self.prev_outputs)
            #outputs = self.full.run({'img': frame})

        if outputs is None:
            return None

        self.prev_outputs = outputs

        self.last_frame = frame

        return outputs
