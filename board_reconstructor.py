from pipeline import PipelineStep
import cv2
import numpy as np
from utils import *
from itertools import combinations, product
from tile_classifier import *

class ReconstructBoard(PipelineStep):
     def process(self, inputs, visualize=False):
          classified_tiles = inputs['classified_tiles']
          locations = inputs['tile_locations']
          if len(classified_tiles) == 0:
               print('No classified tiles!')
               return None

          min_x_index = 10000
          max_x_index = -1
          min_y_index = 10000
          max_y_index = -1
          for (x_index, y_index), tile in zip(locations, classified_tiles):
               if tile is not None:
                    min_x_index = min(min_x_index, x_index)
                    max_x_index = max(max_x_index, x_index)
                    min_y_index = min(min_y_index, y_index)
                    max_y_index = max(max_y_index, y_index)

          size = 64
          height = ((max_y_index - min_y_index) + 1) * size
          width = ((max_x_index - min_x_index) + 1) * size
          board = np.zeros((height, width, 3), dtype=np.uint8)
          for (x_index, y_index), tile in zip(locations, classified_tiles):
               if tile is None:
                    continue
               
               x = (x_index - min_x_index) * size
               y = (y_index - min_y_index) * size
               board[y:y+size, x:x+size] = tile.matched_tile

          outputs = {'board_img': board}

          if visualize:
               outputs['debug_img'] = board

          return outputs


