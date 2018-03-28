from tile_classifier import *

tile_test1 = {
	'1': 'P',
	'2': 'L',
	'3': 'D',
	'4': 'U',
	'5': 'J',
	'6': 'I',
	'7': 'E',
	'8': 'N',
	'9': 'U',
	'11': 'K',
	'12': 'E',
	'13': 'B',
	'14': 'V',
	'15': 'K',
	'16': 'R',
	'17': 'Q',
	'18': 'F',
	'19': 'M',
	'20': 'B',
	'21': 'B',
	'22': 'U',
	'23': 'T',
	'25': 'K',
	'26': 'D',
	'27': 'D',
	'29': 'D',
	'30': 'V',
	'31': 'P',
	'32': 'R',
	'33': 'V',
	'34': 'L',
	'35': 'K',
	'36': 'G',
	'38': 'J',
	'40': 'V',
	'41': 'O',
	'42': 'N',
	'43': 'H',
	'45': 'U',
	'47': 'A',
	'48': 'N',
	'49': 'S',
	'52': 'K',
	'53': 'V',
}

tile_sets = {
	'tile_test1': tile_test1
}

def test_tile_set(set_name, labeled_tiles):
	classifier = TileClassifier()
	tiles = [cv2.imread('data/' + set_name + '/' + name + '.png') for name in labeled_tiles]
	predicted_labels = classifier.process({'tiles': tiles})['tile_labels']
	num_correct = 0
	errors = []
	for tile, predicted, truth in zip(labeled_tiles, predicted_labels, labeled_tiles.values()):
		if predicted == truth:
			num_correct += 1
		else:
			errors.append((tile, predicted, truth))
	total = len(tiles)
	percent_correct = (float(num_correct) / total) * 100
	print('{}: {}/{}, {}%'.format(set_name, num_correct, total, percent_correct))
	print('Errors:', errors)

def test_tile_classifier():
	for set_name, labeled_tiles in tile_sets.items():
		test_tile_set(set_name, labeled_tiles)

test_tile_classifier()