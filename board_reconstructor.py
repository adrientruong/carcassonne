from pipeline import PipelineStep
import cv2
import numpy as np
from utils import *
from itertools import combinations, product
from tile_classifier import *
from default_tiles import *

class ReconstructBoard2(PipelineStep):
     def process(self, inputs, visualize=False):
          classified_tiles = inputs['classified_tiles']
          locations = inputs['tile_locations']
          default_tiles = get_default_tiles()
          size = 64
          def center_of_tile_in_img_p(x_i, y_i):
               x = x_i * size + (size / 2)
               y = y_i * size + (size / 2)
               board_p = np.array([x, y, 1])
               image_point = H_inv.dot(p.reshape(3, 1))
               image_point = np.squeeze(image_point)
               image_point = image_point[:2] / image_point[2]
               image_point = image_point.astype('int64')

               return image_point

          coordinates_to_tiles = {}
          for (x_index, y_index), tile in zip(locations, classified_tiles):
               if tile is None:
                    continue
               
               x = x_index * size
               y = y_index * size
               default_tile = default_tiles[tile.letter].tile_by_rotating(tile.rotation)
               coordinates_to_tiles[(x, y)] = default_tile

          board = CarcBoard(coordinates_to_tiles)
          

class ReconstructBoard(PipelineStep):
     def process(self, inputs, visualize=False):
          classified_tiles = inputs['classified_tiles']
          locations = inputs['tile_locations']
          if len(classified_tiles) == 0 or not any(classified_tiles):
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

          pieces = inputs['pieces']
          BLUE = (255, 0, 0)
          RED = (0, 0, 255)
          YELLOW = (0, 255, 255)
          BLACK = (0, 0, 0)
          # intentional mix order so we can see outline
          COLORS = [BLUE, RED, YELLOW, BLACK]
          H = inputs['H']
          def image_p_to_board_p(x, y):
               board_p = H.dot(np.array([x, y, 1]))
               board_p = board_p[:2] / board_p[2]
               board_p = board_p.astype(np.int32)
               return board_p

          for pieces_of_color, color in zip(pieces, COLORS):
               for c in pieces_of_color:
                    x,y,w,h = cv2.boundingRect(c)
                    p = image_p_to_board_p(x, y)
                    cv2.rectangle(board, (p[0], p[1]), (p[0]+30, p[1]+30), color,2)

          outputs = {'board_img': board}

          if visualize:
               outputs['debug_img'] = board

          return outputs


