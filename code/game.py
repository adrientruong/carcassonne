from itertools import product

from tile import *

class CarcGameState:
	def __init__(self, players):
		self.tiles_by_position = {}
		self.player_pieces_by_position = {}

		self.players = players
		self.current_player_index = 0

		self.player_points = [0 for _ in players]
		self.turn_num = 0

	def after_playing_turn(self, turn):
		new_state = self.copy()
		new_state.add_tile(turn.tile, turn.position)
		if turn.place_piece:
			new_state.add_piece(turn.position, self.current_player_index)
		new_state.resolve_turn(turn)

		new_state.current_player_index += 1
		new_state.current_player_index = new_state.current_player_index % len(self.players)
		new_state.turn_num += 1

		return new_state

	def active_positions(self):
		return self.tiles_by_position.keys()

	def points_of_player_index(self, index):
		return self.player_points[index]

	def potential_points_of_player_index(self, index):
		potential_points = 0
		for position, i in self.player_pieces_by_position.items():
			if i != index:
				continue

			tile = self.tile_at_position(position)

			feature_point_map = self.incomplete_feature_point_map()
			for edges in tile.all_connecting_edges_set():
				edge = edges[0] # doesn't matter which one we pick

				feature = tile.feature_on_edge(edge)
				if feature == TileFeature.GRASS:
					continue

				_, positions, piece_counts = self.check_progress_of_feature(position, edge, [], {})

				max_pieces = max(piece_counts.values())
				owning_player_indices = [i for i, count in piece_counts.items() if count >= max_pieces]

				if i not in owning_player_indices:
					continue

				points_per_tile = feature_point_map[feature]
				points_earned = len(positions) * points_per_tile
				if feature == TileFeature.CITY:
					tiles = [self.tile_at_position(p) for p in positions]
					pennant_bonus = sum([points_per_tile for tile in tiles if tile.has_pennant])
					points_earned += pennant_bonus
				
				potential_points += points_earned
		return potential_points

	def position_adjacent_to_edge(self, p, edge):
		if edge == TileEdge.LEFT:
			return (p[0] - 1, p[1])
		elif edge == TileEdge.TOP:
			return (p[0], p[1] - 1)
		elif edge == TileEdge.RIGHT:
			return (p[0] + 1, p[1])
		elif edge == TileEdge.BOTTOM:
			return (p[0], p[1] + 1)
		else:
			assert False, 'Invalid edge'

	def can_place_tile(self, tile, p):
		for edge in TileEdge.all_edges():
			adj_p = self.position_adjacent_to_edge(p, edge)
			adj_tile = self.tile_at_position(adj_p)
			if adj_tile is None:
				continue

			if adj_tile.feature_on_edge(edge.opposite()) != tile.feature_on_edge(edge):
				return False

		return True

	def required_features_for_placement(self, pos):
		features = {}
		for edge in TileEdge.all_edges():
			adj_pos = self.position_on_edge_of_position(pos, edge)
			adj_tile = self.tile_at_position(adj_pos)
			if adj_tile is not None:
				features[edge.feature_location()] = adj_tile.feature_on_edge(edge.opposite())

		return features

	def add_tile(self, tile, position):
		assert position not in self.tiles_by_position, 'Cannot place tile, tile already there'
		assert self.can_place_tile(tile, position), 'Invalid tile placement! Features dont match'
		self.tiles_by_position[position] = tile

	def add_piece(self, position, player_index):
		assert position in self.tiles_by_position, 'Cannot place piece without tile'
		assert position not in self.player_pieces_by_position, 'Cannot place piece, piece already there'
		self.player_pieces_by_position[position] = player_index

	def remove_piece(self, position):
		assert position in self.tiles_by_position, 'Cannot remove piece without tile'
		assert position in self.player_pieces_by_position, 'Cannot remove piece, piece not there'

		self.player_pieces_by_position[position] = None

	def owning_player_of_position(self, position):
		return self.player_pieces_by_position.get(position, None)

	def add_points_for_player_index(self, points, player_index):
		self.player_points[player_index] += points

	def tile_at_position(self, position):
		return self.tiles_by_position.get(position, None)

	def placeable_positions(self):
		valid_positions = set()
		for position in self.tiles_by_position:
			for adj_pos in self.manhattan_positions_around(position):
				if self.tile_at_position(adj_pos) is None:
					valid_positions.add(adj_pos)
		return valid_positions

	def manhattan_positions_around(self, p):
		left = (p[0] - 1, p[1])
		top = (p[0], p[1] - 1)
		right = (p[0] + 1, p[1])
		bottom = (p[0], p[1] + 1)

		return [left, top, right, bottom]

	def positions_around(self, position):
		positions = []
		for x_offset, y_offset in product(range(-1, 2), range(-1, 2)):
			if x_offset == 0 and y_offset == 0:
				continue
			positions.append((position[0] + x_offset, position[1] + y_offset))

		return positions

	def position_on_edge_of_position(self, position, edge):
		if edge == TileEdge.LEFT:
			return (position[0] - 1, position[1])
		elif edge == TileEdge.TOP:
			return (position[0], position[1] - 1)
		elif edge == TileEdge.RIGHT:
			return (position[0] + 1, position[1])
		elif edge == TileEdge.BOTTOM:
			return (position[0], position[1] + 1)
		else:
			assert False, 'Invalid edge given'

	def completed_feature_point_map(self):
		return {
			TileFeature.CITY: 2,
			TileFeature.ROAD: 1,
			TileFeature.MONASTERY: 1
		}

	def incomplete_feature_point_map(self):
		return {
			TileFeature.CITY: 1,
			TileFeature.ROAD: 1,
			TileFeature.MONASTERY: 1
		}

	def resolve_turn(self, turn):
		points_earned = 0
		feature_point_map = self.completed_feature_point_map()
		for edges in turn.tile.all_connecting_edges_set():
			edge = edges[0] # doesn't matter which one we pick

			feature = turn.tile.feature_on_edge(edge)
			if feature == TileFeature.GRASS:
				continue

			finished, positions, piece_counts = self.check_progress_of_feature(turn.position, edge, [], {})
			if finished:
				if len(piece_counts) == 0:
					# no one owns this
					continue

				max_pieces = max(piece_counts.values())
				owning_player_indices = [i for i, count in piece_counts.items() if count >= max_pieces]

				points_per_tile = feature_point_map[feature]
				points_earned = len(positions) * points_per_tile
				if feature == TileFeature.CITY:
					tiles = [self.tile_at_position(p) for p in positions]
					pennant_bonus = sum([points_per_tile for tile in tiles if tile.has_pennant])
					points_earned += pennant_bonus
				for i in owning_player_indices:
					self.add_points_for_player_index(points_earned, i)

				for p in positions:
					if self.owning_player_of_position(p) is not None:
						self.remove_piece(p)

		for adj_p in [turn.position] + self.positions_around(turn.position):
			tile = self.tile_at_position(adj_p)
			if tile is None:
				continue

			if turn.tile.feature_at_location(FeatureLocation.CENTER) != TileFeature.MONASTERY:
				continue

			# is monastery

			owning_player = self.owning_player_of_position(adj_p)
			if owning_player is None:
				continue

			finished, all_positions = self.check_progress_of_monastery(adj_p)
			if finished:
				points_earned = feature_point_map[TileFeature.MONASTERY] * len(all_positions)
				self.add_points_for_player_index(points_earned, owning_player)

	# Returns tuple (is_finished, all_positions, piece_counts)
	# is_finished is whether the feature is complete
	# all_positions is all the tiles that are part of the feature
	# piece_counts is a dictionary that maps player indices to the number of pieces they have on this feature
	def check_progress_of_feature(self, position, edge, positions, piece_counts):
		if position in positions:
			return True, positions, piece_counts

		tile = self.tile_at_position(position)
		if tile is None:
			return False, None, None

		positions.append(position)
		owning_player = self.owning_player_of_position(position)
		if owning_player is not None:
			if owning_player in piece_counts:
				piece_counts[owning_player] += 1
			else:
				piece_counts[owning_player] = 1

		connecting_edges = tile.connecting_edges(edge)
		completely_finished = True
		for c_edge in tile.connecting_edges(edge):
			adj_position = self.position_on_edge_of_position(position, c_edge)
			finished, _, _ = self.check_progress_of_feature(adj_position,
				c_edge.opposite(), positions, piece_counts)
			if not finished:
				completely_finished = False

		return completely_finished, positions, piece_counts

	def check_progress_of_monastery(self, position):
		neighboring_pos = self.positions_around(position)
		neighboring_tiles = [self.tile_at_position(p) for p in neighboring_pos]
		filled_neighbors = [p for p in neighboring_pos if self.tile_at_position(p) is not None]
		monastery_tiles = filled_neighbors + [position]
		finished = len(monastery_tiles) == 9

		return finished, monastery_tiles

	def copy(self):
		copy = CarcGameState(list(self.players))

		tiles_by_position_copy = {}
		for position, tile in self.tiles_by_position.items():
			tiles_by_position_copy[position] = tile.copy()
		copy.tiles_by_position = tiles_by_position_copy
		copy.player_pieces_by_position = dict(self.player_pieces_by_position)

		copy.current_player_index = self.current_player_index
		copy.player_points = list(self.player_points)

		return copy


class Player:
	def __init__(self, name):
		self.name = name

class Turn:
	def __init__(self, tile, position, place_piece=False):
		self.tile = tile
		self.position = position
		self.place_piece = place_piece
