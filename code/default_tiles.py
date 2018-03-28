from tile import *
import cv2
import string

A = Tile(
	{
		FeatureLocation.LEFT: TileFeature.GRASS,
		FeatureLocation.TOP: TileFeature.GRASS,
		FeatureLocation.RIGHT: TileFeature.GRASS,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
		FeatureLocation.CENTER: TileFeature.MONASTERY,
	},
	cv2.imread('data/tiles/A.png')
)

B = Tile(
	{
		FeatureLocation.LEFT: TileFeature.GRASS,
		FeatureLocation.TOP: TileFeature.GRASS,
		FeatureLocation.RIGHT: TileFeature.GRASS,
		FeatureLocation.BOTTOM: TileFeature.GRASS,
		FeatureLocation.CENTER: TileFeature.MONASTERY,
	},
	cv2.imread('data/tiles/B.png')
)

C = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.CITY,
	},
	cv2.imread('data/tiles/C.png'),
	feature_connections=[TileEdge.all_edges()],
	has_pennant=True
)

D = Tile(
	{
		FeatureLocation.LEFT: TileFeature.GRASS,
		FeatureLocation.TOP: TileFeature.ROAD,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/D.png'),
	feature_connections=[[TileEdge.TOP, TileEdge.BOTTOM]]
)

E = Tile(
	{
		FeatureLocation.LEFT: TileFeature.GRASS,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.GRASS,
		FeatureLocation.BOTTOM: TileFeature.GRASS,
	},
	cv2.imread('data/tiles/E.png')
)

F = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.GRASS,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.GRASS,
	},
	cv2.imread('data/tiles/F.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.RIGHT]],
	has_pennant=True
)

G = Tile(
	{
		FeatureLocation.LEFT: TileFeature.GRASS,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.GRASS,
		FeatureLocation.BOTTOM: TileFeature.CITY,
	},
	cv2.imread('data/tiles/G.png'),
	feature_connections=[[TileEdge.TOP, TileEdge.BOTTOM]]
)

H = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.GRASS,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.GRASS,
	},
	cv2.imread('data/tiles/H.png')
)

I = Tile(
	{
		FeatureLocation.LEFT: TileFeature.GRASS,
		FeatureLocation.TOP: TileFeature.GRASS,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.CITY,
	},
	cv2.imread('data/tiles/I.png')
)

J = Tile(
	{
		FeatureLocation.LEFT: TileFeature.GRASS,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.ROAD,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/J.png'),
	feature_connections=[[TileEdge.RIGHT, TileEdge.BOTTOM]]
)

K = Tile(
	{
		FeatureLocation.LEFT: TileFeature.ROAD,
		FeatureLocation.TOP: TileFeature.ROAD,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.GRASS,
	},
	cv2.imread('data/tiles/K.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP]]
)

L = Tile(
	{
		FeatureLocation.LEFT: TileFeature.ROAD,
		FeatureLocation.TOP: TileFeature.ROAD,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/L.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP, TileEdge.BOTTOM]]
)

M = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.GRASS,
		FeatureLocation.BOTTOM: TileFeature.GRASS,
	},
	cv2.imread('data/tiles/M.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP]],
	has_pennant=True
)

N = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.GRASS,
		FeatureLocation.BOTTOM: TileFeature.GRASS,
	},
	cv2.imread('data/tiles/N.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP]]
)

O = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.ROAD,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/O.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP], [TileEdge.RIGHT, TileEdge.BOTTOM]],
	has_pennant=True
)

P = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.ROAD,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/P.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP], [TileEdge.RIGHT, TileEdge.BOTTOM]]
)

Q = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.GRASS,
	},
	cv2.imread('data/tiles/Q.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP, TileEdge.RIGHT]],
	has_pennant=True
)

R = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.GRASS,
	},
	cv2.imread('data/tiles/R.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP, TileEdge.RIGHT]]
)

S = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/S.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP, TileEdge.RIGHT]],
	has_pennant=True
)

T = Tile(
	{
		FeatureLocation.LEFT: TileFeature.CITY,
		FeatureLocation.TOP: TileFeature.CITY,
		FeatureLocation.RIGHT: TileFeature.CITY,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/T.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.TOP, TileEdge.RIGHT]]
)

U = Tile(
	{
		FeatureLocation.LEFT: TileFeature.GRASS,
		FeatureLocation.TOP: TileFeature.ROAD,
		FeatureLocation.RIGHT: TileFeature.GRASS,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/U.png'),
	feature_connections=[[TileEdge.TOP, TileEdge.BOTTOM]]
)

V = Tile(
	{
		FeatureLocation.LEFT: TileFeature.ROAD,
		FeatureLocation.TOP: TileFeature.GRASS,
		FeatureLocation.RIGHT: TileFeature.GRASS,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/V.png'),
	feature_connections=[[TileEdge.LEFT, TileEdge.BOTTOM]]
)

W = Tile(
	{
		FeatureLocation.LEFT: TileFeature.ROAD,
		FeatureLocation.TOP: TileFeature.GRASS,
		FeatureLocation.RIGHT: TileFeature.ROAD,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/W.png')
)


X = Tile(
	{
		FeatureLocation.LEFT: TileFeature.ROAD,
		FeatureLocation.TOP: TileFeature.ROAD,
		FeatureLocation.RIGHT: TileFeature.ROAD,
		FeatureLocation.BOTTOM: TileFeature.ROAD,
	},
	cv2.imread('data/tiles/X.png')
)

def get_default_tiles():
	tiles = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X]
	tile_map = {}
	letters = string.ascii_uppercase
	for tile, letter in zip(tiles, letters):
		tile_map[letter] = tile
	return tile_map

