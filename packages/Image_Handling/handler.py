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


def find_contours(image):  # former get_letters
    """
    Analyzes an image and returns in a list each individual contour along with its X coordinate
    :param image: Black and white image array (Threshold image).
    :return:List of lists, first element is an image array while the second its X coordinates
    """
    to_analyze = []  # letters will be placed here separately

    image_inv = (255 - image)  # Inverting image colors. Black to white and white to black

    # Retrieving all contours in the image
    contours = cv2.findContours(image_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    print(f"Contours found: {len(contours)}")

    # Analyzing each found contour
    for contour in contours:

        # Removing noise. Analyze contours which area is greater than 1000
        if cv2.contourArea(contour) > 1000:
            # Getting each contour x and y position as well as its width and height
            x, y, w, h = cv2.boundingRect(contour)

            e = 5  # Additional pixels at each of the four sides.
            # Appending each individual letter (rectangle) into a list to perform further analysis
            roi = image[y - e: y + h + e, x - e: x + w + e]
            to_analyze.append([roi, x])  # [individual letter, X position]

    print(f"Letters split: {len(to_analyze)}")
    return to_analyze


def prep_letters(to_analyze):
    """
    Analyzes each letter individually
    :param to_analyze: List of list. First element image array, second X coordinate
    :return: List of sorted letters, each item is an image array of equal height and width, perfect square
    """
    letters_predict = []
    # individual letter, X position. Sorting ascending by X value to_analyze[1]
    letters_sorted = sorted(to_analyze, key=lambda x: x[1])

    # Centering each letter into a square
    for i in range(len(letters_sorted)):

        letter = letters_sorted[i][0]
        h, w = letters_sorted[i][0].shape
        print(letters_sorted[i][0].shape, "Area:", h * w)

        # Centering the letter into a perfect square, same height and width
        if h >= w:
            # if the image is taller than it is wide
            square_size = h
            extra_side_pixels = int((h - w) / 2)

            frame = np.full((square_size, square_size), 255)
            frame[:, 0 + extra_side_pixels: w + extra_side_pixels] = letter

            letters_predict.append(frame)

        elif h < w:
            # if the image is wider than it is tall
            square_size = w
            extra_side_pixels = int((w - h) / 2)
            frame = np.full((square_size, square_size), 255)
            frame[0 + extra_side_pixels: h + extra_side_pixels, :] = letter

            letters_predict.append(frame)

    return letters_predict


def get_word(img_path):
    """
    Function that assemble all previous 4 functions
    :param img_path: Image to analyze.
    :return: List. Each item is an image array ( letters )
    """
    # Exit the code if the image is not found
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found in path: {img_path}")
        exit()

    # Cropping image sides to remove possible noise or unwanted contours.
    cropped = crop_image(img, ratio=10)

    # Turning original image into grayscale. Letters black, background white.
    thresh = black_and_white(cropped, threshold=120)

    # Find letters and dismiss unwanted contours.
    letters = find_contours(thresh)

    # Put letters in order and into a square. This list will be used to feed the model.predict() function
    letters_to_predict = prep_letters(letters)

    return letters_to_predict
