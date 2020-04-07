import packages.Acquisition.terminal_cmd as ap
import packages.Acquisition.acquisition as aq
import packages.Model.model as ml

# Setting up constants.
DATA = "../data/raw"
LETS = list("SINO")
IMG_SIZE = 100
MODEL_PATH = "../data/model"
MODEL_NAME = "model"


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

        # Saving model as a H5 file.
        ml.save_model_h5(model, model_path=MODEL_PATH, model_name=MODEL_NAME)

    elif args.predict:
        # Loading model from a H5 file.
        loaded_model = ml.load_model_h5(model_path=MODEL_PATH, model_name=MODEL_NAME)


if __name__ == "__main__":
    main()

