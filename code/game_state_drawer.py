import cv2
import numpy as np

BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
COLORS = [BLUE, RED, YELLOW, BLACK]

def draw_game_state(state):
    active_positions = state.active_positions()
    min_x_index = 10000
    max_x_index = -1
    min_y_index = 10000
    max_y_index = -1
    for x_index, y_index in active_positions:
        min_x_index = min(min_x_index, x_index)
        max_x_index = max(max_x_index, x_index)
        min_y_index = min(min_y_index, y_index)
        max_y_index = max(max_y_index, y_index)

    size = 64
    height = ((max_y_index - min_y_index) + 1) * size
    width = ((max_x_index - min_x_index) + 1) * size
    board = np.zeros((height, width, 3), dtype=np.uint8)

    for x_index, y_index in active_positions:        
        x = (x_index - min_x_index) * size
        y = (y_index - min_y_index) * size
        tile = state.tile_at_position((x_index, y_index))
        board[y:y+size, x:x+size] = cv2.resize(tile.img, (size, size))

        owning_player_index = state.owning_player_of_position((x_index, y_index))
        if owning_player_index is not None:
            color = COLORS[owning_player_index]
            cv2.rectangle(board, (x+20, y+20), (x+40, y+40), color, 2)

    return board