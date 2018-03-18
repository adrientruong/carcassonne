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

	def feature_location(self):
		if self == TileEdge.LEFT:
			return FeatureLocation.LEFT
		elif self == TileEdge.RIGHT:
			return FeatureLocation.RIGHT
		elif self == TileEdge.TOP:
			return FeatureLocation.TOP
		elif self == TileEdge.BOTTOM:
			return FeatureLocation.BOTTOM

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
			feature = self.feature_on_edge(connection[0])
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
		else:
			assert False, 'Invalid edge'

	def feature_at_location(self, location):
		return self.features.get(location, None)

	def all_connecting_edges_set(self):
		if len(self.feature_connections) > 0:
			connecting_edges_set = []
			remaining_edges = set(TileEdge.all_edges())
			for connection in self.feature_connections:
				for edge in connection:
					remaining_edges.remove(edge)
			for edge in remaining_edges:
				connecting_edges_set.append([edge])
			return connecting_edges_set
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
		edge_list = [TileEdge.LEFT, TileEdge.TOP, TileEdge.RIGHT, TileEdge.BOTTOM]
		d = collections.deque(edge_list)
		d.rotate(-r)
		rotated_edge_list = list(d)

		edge_map = {
			rotated_edge_list[0]: TileEdge.LEFT,
			rotated_edge_list[1]: TileEdge.TOP,
		 	rotated_edge_list[2]: TileEdge.RIGHT,
			rotated_edge_list[3]: TileEdge.BOTTOM,
		}

		location_map = { orig_e.feature_location(): rot_e.feature_location() for orig_e, rot_e in edge_map.items() }
		location_map[FeatureLocation.CENTER] = FeatureLocation.CENTER

		rotated_features = {}
		for location, feature in self.features.items():
			rot_loc = location_map[location]
			rotated_features[rot_loc] = feature

		rotated_feature_connections = []
		for connection in self.feature_connections:
			rot_connection = [edge_map[e] for e in connection]
			rotated_feature_connections.append(rot_connection)

		rotated_img = np.rot90(self.img, k=r)

		return Tile(rotated_features, rotated_img, feature_connections=rotated_feature_connections, has_pennant=self.has_pennant)

	def has_features(self, features):
		for location, feature in features.items():
			if self.feature_at_location(location) != feature:
				return False
		return True

	def copy(self):
		return Tile(self.features, self.img, self.feature_connections, self.has_pennant)

