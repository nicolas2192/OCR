import numpy as np
import matplotlib.pyplot as plt
import cv2

import collections
import os
import random
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


def shuffle(list_to_shuffle):
	"""
	Shuffles a list of list to prevent over-fitting
	:param list_to_shuffle:
	:return: Returns a shuffled list.
	"""
	random.shuffle(list_to_shuffle)
	return list_to_shuffle


def prep_data(training_data, letters_list, pixels=100):
	"""
	Preparing the data that will be used to feed the model
	:param training_data: Training data already shuffled.
	:param letters_list: Letters to analyze. Same list used before to load the images.
	:param pixels: Image width and height. Should be the same value used when loading the images. Default 100.
	:return: X keras arrays and y array ready to feed the model
	"""
	print("Preparing data...")
	X, y = [], []

	# Splitting data into training data (X) and its label or target (y)
	for features, label in training_data:
		X.append(features)
		y.append(label)

	# Converting y list into a categorical array, 0 and 1. It will have as many columns as letters to train
	y = tf.keras.utils.to_categorical(y, len(letters_list))

	# Converting X list into a keras array ready to be used to train the model
	X = np.array(X).reshape(-1, pixels, pixels, 1)  # 1 grayscale, 3 colored images

	# Normalizing training data
	# X = tf.keras.utils.normalize(X, axis=1)
	X = X / 255
	print(f"Training data set (X): {X.shape}, Label (y): {y.shape}")

	return X, y


def model_setup(X, y, letters_list):
	"""
	Setting up the model (layers and compiler) before fitting it.
	:param X: Training data set to train the model.
	:param y: Label corresponding to each element in the training data set.
	:param letters_list: List of letters.
	:return: model instance ready to predict new data.
	"""
	# Model's name
	NAME = f"OCR-CNN-{int(time.time())}"
	print(f"\nModel's name: {NAME}")

	# Model instantiation and layers definition
	model = Sequential()  # instantiating the NN

	# Adding 1st 2D convolution layer
	model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], activation="relu"))  # X.shape[1:] → (100, 100, 1)
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	# Adding 2nd 2D convolution layer
	model.add(Conv2D(64, (3, 3), activation="relu"))  # don't have to specify input shape in additional layers.
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	# Adding 3rd 2D convolution layer
	model.add(Conv2D(64, (3, 3), activation="relu"))  # don't have to specify input shape in additional layers.
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))

	# Flattening before passing it to the dense layer
	model.add(Flatten())

	# Adding output layer, as many units as letters there are, activation softmax
	model.add(Dense(len(letters_list), activation="softmax"))

	# Setting up tensorboard logs
	tb = TensorBoard(log_dir=f"data/logs/{NAME}")

	# Model parameters
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	# Model summary
	model.summary()
	print("")

	# Model fitting
	hist = model.fit(X, y, batch_size=32, epochs=6, validation_split=0.2, callbacks=[tb])
	print("")

	return model, hist


def plot_performance(hist, model_path="data/model", model_name="model"):
	"""
	Creates a png Hile with two charts. Loss and Accuracy
	:param hist: History object from the model.fit() function
	:param model_path: Path where the PNG image will be saved to. Default: data/model
	:param model_name: Performance chart name
	:return: PNG file
	"""
	fig, axes = plt.subplots(1, 2, figsize=(15, 5))
	fig.suptitle("Model Performance Summary", fontsize=18)

	x_ticks = np.arange(1, len(hist.history["loss"]) + 1)

	label_font = {"fontname": "Arial", "fontsize": 13}
	title_font = {"fontname": "Arial", "fontsize": 16}

	axes[0].set_title("Loss", fontdict=title_font)
	axes[0].set_xlabel("Epoch", fontdict=label_font)
	axes[0].set_xticks(np.arange(len(x_ticks)))
	axes[0].set_xticklabels(x_ticks)
	axes[0].plot(hist.history["loss"], label="Train")
	axes[0].plot(hist.history["val_loss"], label="Test")
	axes[0].grid(True, color="black", linewidth=0.1)
	axes[0].legend()

	axes[1].set_title("Accuracy", fontdict=title_font)
	axes[1].set_xlabel("Epoch", fontdict=label_font)
	axes[1].set_xticks(np.arange(len(x_ticks)))
	axes[1].set_xticklabels(x_ticks)
	axes[1].plot(hist.history["accuracy"], label="Train")
	axes[1].plot(hist.history["val_accuracy"], label="Test")
	axes[1].grid(True, color="black", linewidth=0.1)
	axes[1].legend(loc=4)

	try:
		plt.savefig(os.path.join(model_path, f"{model_name}.png"))
		print(f"Model performance chart was successfully saved to: {model_path}/{model_name}.png")
	except Exception as e:
		print("Error saving model chart")


def model_performance(model, X, y):
	"""
	Prints model performance. Whether trained or loaded.
	:param model: Model to check
	:param X: Training data set
	:param y: Corresponding label
	:return: Prints model performance. Returns overall loss and accuracy
	"""
	print("Overall Loss and Accuracy")
	loss, acc = model.evaluate(X, y, verbose=1)
	print("")
	return loss, acc


def save_model_h5(model, model_path="data/model", model_name="model"):
	"""
	Saves model to a .h5 file
	:param model: Model to be saved as a .h5 file in the specified path
	:param model_path: Path where the .h5 file model will be saved to. Default: data/model
	:param model_name: Model's name. Default: model
	:return: Saves the model to the specified path under the specified name
	"""
	model.save(os.path.join(model_path, f"{model_name}.h5"))
	print(f"Model saved to: {model_path}/{model_name}.h5")


def load_model_h5(model_path="data/model", model_name="model"):
	"""
	Loads a previously saved model in a .h5 file
	:param model_path: Path where the .h5 file model will be read from. Default: data/model
	:param model_name: Model's name. Default: model
	:return: Loaded model
	"""
	try:
		# Loading H5 file
		print("Model ->", os.path.join(model_path, f"{model_name}.h5"))
		loaded_model = load_model(os.path.join(model_path, f"{model_name}.h5"))
		print("\nModel loaded successfully!")
		return loaded_model
	except Exception as e:
		print("Model->", os.path.join(model_path, f"{model_name}.h5"))
		print("\nModel couldn't be loaded. Did you import keras load_model function?")
		exit()


def average_predict(keras_array, m, letters_list, fakes=7):
	"""
	:param keras_array: Original image in keras array format
	:param m: model to be used to predict the letters
	:param letters_list: Letters spectrum to choose from
	:param fakes: Number of randomly generated images
	:return: Returns a list of randomly generated keras arrays from the original image (keras_array)
	"""
	datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1, height_shift_range=0.1)
	artificial_letters = []
	for f in range(fakes - 1):
		fake = datagen.flow(keras_array, batch_size=1)
		pred = m.predict(fake)
		pred_index = np.argmax(pred)
		artificial_letters.append(letters_list[pred_index])

	c = collections.Counter(artificial_letters)

	return c.most_common(1)[0][0], f"{c.most_common(1)[0][1]}/{fakes}"


def read_letters(model, letters_to_predict, letters_list, pixels=100):
	"""
	:param model: Model previously loaded into memory to predict image arrays.
	:param letters_to_predict: List of image arrays, sorted, each array represents a letter.
	:param letters_list: Letters to analyze. Same list used before to load the images.
	:param pixels: Image width and height. Should be the same value used when loading the images. Default 100.
	:return: Word in a string object
	"""
	word = []

	fig, axes = plt.subplots(1, len(letters_to_predict), figsize=(16, 4))
	label_font = {"fontname": "Arial", "fontsize": 18}

	# Analyzing each image array separately
	for index in range(len(letters_to_predict)):
		# Predicting each letter
		let = letters_to_predict[index].astype("float32")  # array must be float32 for the resize function to work
		let = cv2.resize(let, (pixels, pixels))
		let = cv2.GaussianBlur(let, (5, 5), 0)  # removes high frequency noise
		let = np.array(let).reshape(-1, pixels, pixels, 1)

		# pred = model.predict(let)
		# pred_index = np.argmax(pred)
		# word.append(letters_list[pred_index])
		# print(f"Letter predicted: {letters_list[pred_index]} - Precision: {pred[0][pred_index]}")
		#
		# # Printing each letter into a matplotlib axis
		# axes[index].imshow(letters_to_predict[index], cmap="gray")
		# axes[index].set_title(letters_list[pred_index], fontdict=label_font)
		# axes[index].set_xlabel(pred[0][pred_index], fontdict=label_font)
		# axes[index].set_xticks([])
		# axes[index].set_yticks([])

		# Average method, predicts randomly generated images and returns the most common predicted letter
		avg_letter, acc = average_predict(keras_array=let, m=model, letters_list=letters_list, fakes=7)

		word.append(avg_letter)
		print(f"Predicted Letter: {avg_letter} - Precision: {acc}")

		# Printing each letter into a matplotlib axis
		axes[index].imshow(letters_to_predict[index], cmap="gray")
		axes[index].set_title(avg_letter, fontdict=label_font)
		axes[index].set_xlabel(acc, fontdict=label_font)
		axes[index].set_xticks([])
		axes[index].set_yticks([])

	word = "".join(word)

	fig.suptitle(word, fontsize=20, y=0.9)
	plt.savefig(f"data/prediction/{word}.png")

	return word
