from itertools import product

from game import *
from default_tiles import *
from game_state_drawer import *

from utils import *

def show_state(state):
    for player, points in zip(state.players, state.player_points):
        print('{}: {}'.format(player.name, points))
    print('-' * 80)

    board_img = draw_game_state(state)
    cv2.imshow('board', board_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


tiles = get_default_tiles()

def test_player_index_changes_after_turns():
    player1 = Player('Adrien')
    player2 = Player('Kyle')
    players = [player1, player2]
    state = CarcGameState(players)

    turn1 = Turn(tiles['B'], (0, 0), place_piece=True)

    state = state.after_playing_turn(turn1)
    assert state.current_player_index == 1, 'Expected index 1 got {}'.format(state.current_player_index)

    turn2 = Turn(tiles['E'], (0, -1))

    state = state.after_playing_turn(turn2)
    assert state.current_player_index == 0, 'Expected index 0 got {}'.format(state.current_player_index)

def test_completed_monastery():
    player1 = Player('Adrien')
    player2 = Player('Kyle')
    players = [player1, player2]
    state = CarcGameState(players)

    # place initial monastery
    turn = Turn(tiles['B'], (0, 0), place_piece=True)
    state = state.after_playing_turn(turn)

    # place tiles around
    for x, y in product(range(-1, 2), range(-1, 2)):
        if x == 0 and y == 0:
            continue

        turn = Turn(tiles['B'], (x, y))
        state = state.after_playing_turn(turn)

    #show_state(state)

    assert state.points_of_player_index(0) == 9

def test_basic_completed_city1():
    player1 = Player('Adrien')
    player2 = Player('Kyle')
    players = [player1, player2]
    state = CarcGameState(players)

    turn1 = Turn(tiles['E'], (0, 0), place_piece=True)
    state = state.after_playing_turn(turn1)

    assert state.potential_points_of_player_index(0) == 1

    turn2 = Turn(tiles['E'].tile_by_rotating(2), (0, -1))
    state = state.after_playing_turn(turn2)

    #show_state(state)

    assert state.points_of_player_index(0) == 4

def test_basic_completed_city2():
    player1 = Player('Adrien')
    player2 = Player('Kyle')
    players = [player1, player2]
    state = CarcGameState(players)

    turn1 = Turn(tiles['E'], (0, 0))
    state = state.after_playing_turn(turn1)

    turn2 = Turn(tiles['E'].tile_by_rotating(2), (0, -1), place_piece=True)
    state = state.after_playing_turn(turn2)

    #show_state(state)

    assert state.points_of_player_index(1) == 4

def test_basic_completed_city3():
    player1 = Player('Adrien')
    player2 = Player('Kyle')
    players = [player1, player2]
    state = CarcGameState(players)

    turn1 = Turn(tiles['L'], (0, 0))
    state = state.after_playing_turn(turn1)

    turn2 = Turn(tiles['G'].tile_by_rotating(1), (1, 0), place_piece=True)
    state = state.after_playing_turn(turn2)

    assert state.points_of_player_index(1) == 0

    turn3 = Turn(tiles['E'].tile_by_rotating(1), (2, 0))
    state = state.after_playing_turn(turn3)

    assert state.points_of_player_index(1) == 6, 'expected 8 got: {}'.format(state.points_of_player_index(1))

def test_placeable_positions():
    player1 = Player('Adrien')
    player2 = Player('Kyle')
    players = [player1, player2]
    state = CarcGameState(players)

    turn1 = Turn(tiles['E'], (0, 0))
    state = state.after_playing_turn(turn1)

    turn2 = Turn(tiles['E'].tile_by_rotating(2), (0, -1), place_piece=True)
    state = state.after_playing_turn(turn2)

    positions = [
        (0, -2),
        (1, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
        (-1, -1)
    ]
    placeable_positions = state.placeable_positions()
    for p in positions:
        assert p in placeable_positions, 'expected {} in placeable positions'.format(p)


test_player_index_changes_after_turns()
test_completed_monastery()
test_basic_completed_city1()
test_basic_completed_city2()
test_basic_completed_city3()
test_placeable_positions()

print('All gucci!')

