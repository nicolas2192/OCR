import cv2
import numpy as np


def black_and_white(image, threshold=120):
    """
    Turns an images into grayscale.
    :param image: Image array to convert.
    :param threshold: black and white threshold. Ranges from 0 (black) to 255 (white). Default 120.
    :return: Input image in a black gray scale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((2, 2))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closing


def crop_image(image, ratio=10):
    """
    Removes side noise by cropping the image.
    :param image: Original image to be cropped.
    :param ratio: Cropping ratio. The higher the number, the smaller the ratio. Default 10.
    :return: Cropped image.
    """
    h, w, *_ = image.shape
    c = image[int(h/ratio): int(h*(ratio-1)/ratio), int(w/ratio): int(w*(ratio-1)/ratio)]  # [y:h,x:w]
    return c
