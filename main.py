import packages.Acquisition.terminal_cmd as ap
import packages.Acquisition.acquisition as aq
import packages.Model.model as ml
import packages.Image_Handling.handler as hd
import packages.Dictionary.meaning as rp
from silence_tensorflow import silence_tensorflow

# Setting up constants.
DATA = "data/alphabet"
LETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMG_SIZE = 100
MODEL_PATH = "data/model"
#MODEL_NAME = f"model_{len(LETS)}_40"
MODEL_NAME = "model_full26_800"

silence_tensorflow()


def main():
    # Argparse
    args = ap.terminal_parser()

    if args.train:
        # Loads image letters
        letters = aq.load_letters(DATA, LETS, IMG_SIZE)
        aq.check_loaded_data(letters, LETS)

        # Shuffles data to prevent over-fitting
        shuffled = ml.shuffle(letters)

        # Splits data into two arrays, X and y, to be used later on to feed the model.
        X, y = ml.prep_data(shuffled, LETS, IMG_SIZE)

        # Setting up the model and fitting it
        model, hist = ml.model_setup(X, y, LETS)

        # Calculating and printing overall performance
        loss, acc = ml.model_performance(model, X, y)

        # Plotting model's performance
        ml.plot_performance(hist, model_path=MODEL_PATH, model_name=MODEL_NAME)

        # Saving model as a H5 file.
        ml.save_model_h5(model, model_path=MODEL_PATH, model_name=MODEL_NAME)

    if args.predict:
        # Loading model from a H5 file.
        loaded_model = ml.load_model_h5(model_path=MODEL_PATH, model_name=MODEL_NAME)

        # Reading image and splitting its letters into image arrays
        letters_to_predict = hd.get_word(args.image)

        # Feeding the model the image arrays (letters)
        predicted_word = ml.read_letters(loaded_model, letters_to_predict, LETS, IMG_SIZE)
        print(f"\nYour word is: {predicted_word}\n")

        if args.search:
            definition = rp.meaning(predicted_word)
            print(f"MEANING:\n{definition}")

    if args.train is False and args.predict is False:
        print('Type "python main.py -h" on your terminal to open the help menu')


if __name__ == "__main__":
    main()
