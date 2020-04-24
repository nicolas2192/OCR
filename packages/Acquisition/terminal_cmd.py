import argparse


def terminal_parser():
	parser = argparse.ArgumentParser(description="Reads a word from an image and turns it into a string.")
	parser.add_argument("-t", "--train", action="store_true",
						help="If called, loads data from data folder, trains the model and saves it as a h5 file.")
	parser.add_argument("-p", "--predict", action="store_true",
						help="If called, loads a previously trained model and analyses whatever image was passed using the -i flag.")
	parser.add_argument("-i", "--image", type=str, metavar="", default="data/test/python.jpeg",
						help="Image to analyse. Example images can be found at data/test. This flag must be used along with predict.")
	parser.add_argument("-s", "--search", action="store_true",
						help="If called, looks up the word in the Oxford dictionary returning its meaning. Use this flag alongside predict.")

	return parser.parse_args()


def str2bool(val):
	if isinstance(val, bool):
		return val
	if val.lower() in ('yes', 'true', 't', 'y', '1', 'si', 's'):
		return True
	elif val.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
