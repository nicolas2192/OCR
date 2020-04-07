import os
import cv2


from packages.Image_Handling.handler import black_and_white


def load_letters(path, letters_list, img_size=100):
	"""
	Loading, resizing and labeling all letters
	:param path: Path where raw letters are placed. There should be one folder per letter.
	:param letters_list: Letter to analyze. Letters in this list and in the folder should be the same but its not mandatory
	:param img_size: Image height and width. Default 100 pixels.
	:return: Returns a list of lists where the first element in the inner list is an image array and the second its label.
	"""
	print("Loading data...")
	letters = []
	# Looping through each letter folder
	for let in letters_list:
		images = os.listdir(os.path.join(path, let))
		# Looping through each image within a specific letter folder
		for pic in images:
			# Reading, resizing and turning the image into grayscale
			img = cv2.imread(os.path.join(path, let, pic))
			resized = cv2.resize(img, (img_size, img_size))
			thresh = black_and_white(resized, 120)

			# Saving letter image array and target into a list
			letters.append([thresh, letters_list.index(let)])

	print(f"Total images: {len(letters)}")
	return letters


def check_loaded_data(loaded_letters_list, individual_letters_list):
	"""
	Checking how many images were read per letter
	:param loaded_letters_list: load_letters() list
	:param individual_letters_list: Same letters list passed as argument in the load_letters() function.
	:return: Prints how many images were read per letter.
	"""
	y = [target[1] for target in loaded_letters_list]
	res = [[letter, y.count(individual_letters_list.index(letter))] for letter in individual_letters_list]
	print("Letters break down:\n", res)
