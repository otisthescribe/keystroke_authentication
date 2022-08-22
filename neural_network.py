import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

BLOCK_SIZE = 60
USERS = 400


def load_model_from_dir(directory: str = "./model") -> (Sequential, np.ndarray):
    """
    Load keras model from directory passed in argument.
    If no value is passed, the default "./model" directory is used.

    :param directory: String indicating directory with neural network model
    :return: Tuple with neural network model and central vector
    """
    # CHECK IF THE FOLDER EXISTS
    model = load_model(directory)
    with open(directory + "/central_vector.pickle", "rb") as file:
        central_vector = pickle.load(file)
    return model, central_vector


def check_score(sample: np.ndarray, template: np.ndarray) -> float:
    """
    Checking the cosine similarity (which is a biometric score) between
    template and sample vectors.

    :param sample: First vector
    :param template: Second vector
    :return: Float value indicating biometric score
    """
    # Transforming vectors so they are in proper dimensions
    a = np.expand_dims(sample, axis=0)
    b = np.expand_dims(template, axis=0)
    score = cosine_similarity(a, b)
    # score is formatted as [[float]] so return just the value
    return score[0][0]


def read_data():

    with open("./data/train_user_data.pickle", 'rb') as file:
        train_data = pickle.load(file)

    with open("./data/eval_user_data.pickle", 'rb') as file:
        eval_data = pickle.load(file)

    return train_data, eval_data


def prepare_data(train_data: dict):
    """
    Prepare data for model training. Divide data into train and valid.

    :param train_data: dictionary with training data
    :return: tuple of six arrays
    """
    X = []
    Y = []

    for person_id in train_data.keys():
        for sample in train_data[person_id]:
            X.append(sample)
            Y.append(person_id)

    X = np.array(X)
    Y = np.array(Y)

    Y_oneshot = to_categorical(Y, num_classes=USERS)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y_oneshot, test_size=0.3, random_state=123
    )
    return X, Y, X_train, X_valid, Y_train, Y_valid


def create_model(X: np.ndarray, X_train: np.ndarray, X_valid: np.ndarray, Y_train: np.ndarray,
                 Y_valid: np.ndarray, ) -> (Sequential, np.ndarray, object):
    """
    Create keras model from training data and evaluate it using valid data.

    :param X: array with everyone's data in one block
    :param X_train: array with training data
    :param X_valid: array with valid data
    :param Y_train: array for output of training data (from model)
    :param Y_valid: array for output of valid data (from model)
    :return: model, central_vector, history data
    """
    # THIS PART NEEDS TO BE REVISED - HOW MANY NEURONS AND HOW MANY LAYERS
    model = Sequential()
    model.add(Dense(units=BLOCK_SIZE, input_dim=BLOCK_SIZE, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=400, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=USERS, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # batch size indicates the number of observations to calculate before updating the weights
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=128, batch_size=16)
    vector_probes = model.predict(X)
    central_vector = np.mean(vector_probes, axis=0)

    return model, central_vector, history


def enroll_users(model: Sequential, eval_data: dict, central_vector: np.ndarray) -> (dict, dict):
    """
    Create a biometric template for every user in a dictionary.

    :param model: neural network model
    :param eval_data: dictionary with user's evaluation data
    :param central_vector: array with vector from everyone's data
    :return: two dictionaries with enroll vector and test vectors for every user
    """
    enroll = {}  # klucz = osoba ; wartosc = template
    test = {}  # klucz = osoba ; wartosc = tab[5]

    for person_id in eval_data.keys():

        SEP = (len(eval_data[person_id]) * 2) // 3

        # ENROLLMENT VECTOR
        temp = eval_data[person_id][:SEP]
        output = model.predict(np.array(temp))
        out_vector = np.mean(output, axis=0)
        enroll[person_id] = np.subtract(out_vector, central_vector)
        print(enroll[person_id])
        print(central_vector)

        # TEST VECTORS
        test_vectors = []
        temp = eval_data[person_id][SEP:]
        output = model.predict(np.array(temp))
        out_vector = np.mean(output, axis=0)
        test_vectors.append(np.subtract(out_vector, central_vector))
        # for sample in output:
        #     # out_vector = np.mean(output, axis=0)
        #     test_vectors.append(np.subtract(sample, central_vector))
        test[person_id] = np.array(test_vectors)

    return enroll, test


def cross_evaluate(enroll: dict, test: dict) -> (np.ndarray, np.ndarray):
    """
    Cross evaluate accuracy of biometric templates with the same user's test vectors
    and with everyone else's test vectors.
    :param enroll: dictionary with enroll vectors
    :param test: dictionary with test vectors
    :return: two numpy arrays with true positives and true negatives
    """
    confidence_TP_MLP = []
    confidence_TN_MLP = []
    for userA in enroll.keys():
        userA_model = np.expand_dims(enroll[userA], axis=0)
        # testing with the same user's test vectors
        for t in test[userA]:
            temp = np.expand_dims(t, axis=0)
            a = cosine_similarity(userA_model, temp)
            confidence_TP_MLP.append(a)
        # testing with other users' test vectors
        for userB in test.keys():
            if userB != userA:
                for t in test[userB]:
                    temp = np.expand_dims(t, axis=0)
                    b = cosine_similarity(userA_model, temp)
                    confidence_TN_MLP.append(b)

    confidence_TP_MLP = np.squeeze(np.array(confidence_TP_MLP))
    confidence_TN_MLP = np.squeeze(np.array(confidence_TN_MLP))

    # save these two array into file for future use

    return confidence_TP_MLP, confidence_TN_MLP


def confidence_figure(confidence_TP_MLP: np.ndarray, confidence_TN_MLP: np.ndarray) -> None:
    """
    Draw two figures and save them into files for future use.

    :param confidence_TP_MLP: array with true positive attempts
    :param confidence_TN_MLP: array with true negative attempts
    """
    # Number of true negatives vs number of true positives
    plt.figure()
    n_TP, bins_TP, patches_TP = plt.hist(confidence_TP_MLP, alpha=0.5, bins=200)
    n_TN, bins_TN, patches_TN = plt.hist(confidence_TN_MLP, alpha=0.5, bins=200)
    plt.legend(["score True Positive", "score True Negative"])
    plt.xlim([-1, 1])
    plt.xlabel("score")
    plt.grid()
    plt.savefig("./plots/confidence_TP_TN.png")
    plt.show()

    # Probability of true negatives based on the threshold
    plt.figure()
    plt.plot(bins_TP[1:], np.cumsum(n_TP) / np.sum(n_TP))
    plt.plot(bins_TN[1:], 1 - (np.cumsum(n_TN) / np.sum(n_TN)))
    plt.grid()
    plt.xlabel("threshold")
    plt.ylabel("probability")
    plt.savefig("./plots/threshold_probability.png")
    plt.show()


def model_accuracy_figure(history) -> None:
    """
    Draw two figures about the accuracy and loss of model during training
    :param history: historical data of model training
    """
    # Model accuracy
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.savefig("./plots/model_accuracy.png")
    plt.show()

    # Model loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.savefig("./plots/model_loss.png")
    plt.show()


def save_model(model: Sequential, directory: str = "model") -> None:
    """
    Save model to ./model folder
    :param model: keras model
    :param directory: string with directory name to store model in
    """
    model.save(directory)


def save_central_vector(central_vector):
    """
    Save central vector for the future use
    :param central_vector: 1 x 41 matrix representing central vector
    """
    with open("model/central_vector.pickle", "wb") as file:
        pickle.dump(central_vector, file, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    """
    Main function to create neural network model and train it with
    test data. Then the evaluation is performed and figures are drawn.
    Model and a central vector are saved to ./model folder.
    """
    # Divide test data into eval and train
    eval_data, train_data = read_data()
    # Prepare data for the model
    X, Y, X_train, X_valid, Y_train, Y_valid = prepare_data(eval_data)
    # Create model and central vector
    model, central_vector, history = create_model(X, X_train, X_valid, Y_train, Y_valid)
    # Create test users' enroll templates and test samples
    enroll, test = enroll_users(model, eval_data, central_vector)
    # Evaluate the model using cross evaluation (every user with everyone)
    conf_TP, conf_TN = cross_evaluate(enroll, test)
    # Draw figures representing confidence and save data
    confidence_figure(conf_TP, conf_TN)
    model_accuracy_figure(history)
    save_model(model)
    save_central_vector(central_vector)


if __name__ == "__main__":
    # If the program is run directly (not just by importing function)
    # run the main() function
    main()
