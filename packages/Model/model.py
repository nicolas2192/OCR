import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

import random
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
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
	X = tf.keras.utils.normalize(X, axis=1)
	print(X.shape, y.shape)

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
	print(NAME)

	# Model instantiation and layers definition
	model = Sequential()  # instantiating the NN

	# Adding first 2D convolution layer
	model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], activation="relu"))  # X.shape[1:] → (100, 100, 1)
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Adding second 2D convolution layer
	model.add(Conv2D(64, (3, 3), activation="relu"))  # don't have to specify input shape in additional layers.
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Adding third 2D convolution layer
	model.add(Conv2D(64, (3, 3), activation="relu"))  # don't have to specify input shape in additional layers.
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Flattening before passing it to the dense layer
	model.add(Flatten())

	# Adding output layer, as many units as letters there are, activation softmax
	model.add(Dense(len(letters_list), activation="softmax"))

	# Setting up tensorboard logs
	tb = TensorBoard(log_dir=f"../data/logs/{NAME}")

	# Model parameters
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

	# Model summary
	model.summary()

	# Model fitting
	hist = model.fit(X, y, batch_size=16, epochs=10, validation_split=0.2, callbacks=[tb])

	return model, hist


def model_performance(model, X, y):
	"""
	Prints model performance. Whether trained or loaded.
	:param model: Model to check
	:param X: Training data set
	:param y: Corresponding label
	:return: Prints model performance. Returns overall loss and accuracy
	"""
	loss, acc = model.evaluate(X, y, verbose=1)
	print(f"Loss: {round(loss, 2)}, Accuracy: {round(acc, 2)}")
	return loss, acc


def save_model_h5(model, model_path="data/model", model_name="model"):
	"""
	Saves model to a .h5 file
	:param model: Model to be saved as a .h5 file in the specified path
	:param model_path: Path where the .h5 file model will be saved to. Default: ../data/model
	:param model_name: Model's name. Default: model
	:return: Saves the model to the specified path under the specified name
	"""
	model.save(os.path.join(model_path, f"{model_name}.h5"))
	print(f"Saved model to {model_path}/{model_name}")


def load_model_h5(model_path="data/model", model_name="model"):
	"""
	Loads a previously saved model in a .h5 file
	:param model_path: Path where the .h5 file model will be read from. Default: ../data/model
	:param model_name: Model's name. Default: model
	:return: Loaded model
	"""
	try:
		# Loading H5 file
		loaded_model = load_model(os.path.join(model_path, f"{model_name}.h5"))
		print(f"Model loaded successfully -> {model_name}.h5")
		return loaded_model
	except Exception as e:
		print("Model couldn't be loaded")
		exit()
