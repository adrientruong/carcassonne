from enum import Enum

class TileSideType(Enum):
	RIVER = 1,
	ROAD,
	GRASS,
	CITY,

class TileSide(Enum):
	LEFT = 1,
	RIGHT,
	ABOVE,
	BELOW

	def opposite(self):
		if self == LEFT:
			return RIGHT
		elif self == RIGHT:
			return LEFT
		elif self == ABOVE:
			return BELOW
		elif self == BELOW:
			return ABOVE

class Tile:
	def __init__(self, has_pennant, left, top, right, bottom):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom.
		self.has_pennant = hasPennant

	def side_type(self, direction):
		if direction == LEFT:
			return self.left
		elif direction == RIGHT:
			return self.right
		elif direction == ABOVE:
			return self.top
		elif direction == BELOW:
			return self.bottom

	def can_place(self, position, tile):
		return self.side_type(position.opposite()) == tile.side_type(position)
