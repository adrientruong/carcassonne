from enum import Enum

class TileFeature(Enum):
	NONE = 0,
	RIVER,
	ROAD,
	GRASS,
	CITY

class FeatureLocation(Enum):
	CENTER = 1,
	LEFT,
	RIGHT,
	BOTTOM,
	TOP

class TileSide(Enum):
	LEFT = 1,
	RIGHT,
	TOP,
	BOTTOM

	def opposite(self):
		if self == LEFT:
			return RIGHT
		elif self == RIGHT:
			return LEFT
		elif self == TOP:
			return BOTTOM
		elif self == BOTTOM:
			return TOP
		elif self == CENTER:
			assert("Opposite of center doesn't make sense!")

class Tile:
	def __init__(self, left, top, right, bottom, image, has_pennant=False):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom
		self.image = image
		self.has_pennant = hasPennant

	def edge_feature(self, direction):
		if direction == TileSide.LEFT:
			return self.left
		elif direction == TileSide.RIGHT:
			return self.right
		elif direction == TileSide.TOP:
			return self.top
		elif direction == TileSide.BOTTOM:
			return self.bottom

	def can_place(self, position, tile):
		return self.edge_feature(position) == tile.edge_feature(position.opposite())
