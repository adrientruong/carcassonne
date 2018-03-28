class CarcBoard:
	def __init__():
		self.tiles_by_position = {}
		self.player_pieces_by_position = {}


	def add_tile(self, tile, position):
		assert(position not in self.tiles_by_position, 'Cannot place tile, tile already there')
		self.tiles_by_position[position] = tile

	def add_piece(self, position, player_index):
		assert(position in self.tiles_by_position, 'Cannot place piece without tile')
		assert(position not in self.player_pieces_by_position)
		self.player_pieces_by_position[position] = player_index

	def tile_at_position(self, position):
		return self.tiles_by_position.get(position, None)

	def copy(self):
		tiles_by_position_copy = {}
		for position, tile in self.tiles_by_position.items():
			tiles_by_position_copy[position] = tile.copy()

		copy = CarcBoard()
		copy.tiles_by_position = tiles_by_position_copy
		copy.player_pieces_by_position = dict(self.player_pieces_by_position)
		
		return copy