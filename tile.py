from enum import Enum
import collections
import numpy as np

class TileFeature(Enum):
	RIVER = 1,
	ROAD = 2,
	GRASS = 3,
	CITY = 4,
	MONASTERY = 5

class FeatureLocation(Enum):
	LEFT = 1,
	TOP = 2,
	RIGHT = 3,
	BOTTOM = 4,
	CENTER = 5

class TileEdge(Enum):
	LEFT = 1,
	TOP = 2,
	RIGHT = 3
	BOTTOM = 4

	def all_edges():
		return [TileEdge.LEFT, TileEdge.RIGHT, TileEdge.TOP, TileEdge.BOTTOM]

	def opposite(self):
		if self == TileEdge.LEFT:
			return TileEdge.RIGHT
		elif self == TileEdge.RIGHT:
			return TileEdge.LEFT
		elif self == TileEdge.TOP:
			return TileEdge.BOTTOM
		elif self == TileEdge.BOTTOM:
			return TileEdge.TOP

class Tile:
	def __init__(self, features, img, feature_connections=[], has_pennant=False):
		self.features = features
		self.img = img
		for connection in feature_connections:
			feature = self.feature_on_edge(feature_connections[0])
			for edge in connection:
				assert self.feature_on_edge(edge) == feature, 'Connections are not of same type'
		self.feature_connections = feature_connections
		self.has_pennant = has_pennant

	def feature_on_edge(self, edge):
		if edge == TileEdge.LEFT:
			return self.features[FeatureLocation.LEFT]
		elif edge == TileEdge.TOP:
			return self.features[FeatureLocation.TOP]
		elif edge == TileEdge.RIGHT:
			return self.features[FeatureLocation.RIGHT]
		elif edge == TileEdge.BOTTOM:
			return self.features[FeatureLocation.BOTTOM]

	def feature_at_location(self, location):
		return self.features.get(location, None)

	def all_connecting_edges_set(self):
		if len(self.feature_connections) > 0:
			return self.feature_connections
		else:
			return [[edge] for edge in TileEdge.all_edges()]

	def connecting_edges(self, edge):
		for connection in self.feature_connections:
			if edge in connection:
				return connection
		return [edge]

	def can_place(self, edge, tile):
		return self.edge_feature(edge) == tile.edge_feature(edge.opposite())


	def tile_by_rotating(self, r):
		features_list = [self.features[FeatureLocation.LEFT],
						self.features[FeatureLocation.TOP],
						self.features[FeatureLocation.RIGHT],
						self.features[FeatureLocation.BOTTOM]]
		d = collections.deque(features_list)
		d.rotate(-r)
		rotated_features_list = list(d)

		rotated_features = dict(self.features)
		rotated_features[FeatureLocation.LEFT] = rotated_features_list[0]
		rotated_features[FeatureLocation.TOP] = rotated_features_list[1]
		rotated_features[FeatureLocation.RIGHT] = rotated_features_list[2]
		rotated_features[FeatureLocation.BOTTOM] = rotated_features_list[3]

		rotated_img = np.rot90(self.img, k=r)

		return Tile(rotated_features, rotated_img, has_pennant=self.has_pennant)

	def copy(self):
		return Tile(self.features, self.img, self.feature_connections, self.has_pennant)

