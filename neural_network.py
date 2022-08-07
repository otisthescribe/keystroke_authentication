import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    """
    Read data from filename and divide it into training and
    evaluation data.
    Training data consists of 41 people's samples.
    Evaluation data consists of 10 people's samples.

    :return: tuple of two dictonaries
    """
    # make it universal, so that other file with data will work too
    with open(f'./freeText/free_text_data.pickle', 'rb') as file:
        users = pickle.load(file)
        train_data = {k: v for k, v in users.items() if k in range(400)}
        eval_data = {k: v for k, v in users.items() if k in range(400, 500)}

    return train_data, eval_data


def prepare_data(train_data: dict) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Prepare data for model training. Divide data into train and valid.

    :param train_data: dictionary with training data
    :return: tuple of six arrays
    """
    X = []
    Y = []
    block_size = 60

    for person_id in train_data.keys():
        blocks = 0
        while (blocks + 1) * block_size < len(train_data[person_id]):
            X.append(train_data[person_id][blocks * block_size: (blocks + 1) * block_size])
            Y.append(person_id)
            blocks += 1

    X = np.array(X)
    Y = np.array(Y)
    users_num = len(train_data.keys())
    Y_oneshot = to_categorical(Y, num_classes=users_num)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y_oneshot, test_size=0.2, random_state=123
    )
    return X, Y, X_train, X_valid, Y_train, Y_valid, users_num


def feature_scaling(X_train, X_valid):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_valid = sc.fit_transform(X_valid)
    return X_train, X_valid


def create_model(X: np.ndarray, X_train: np.ndarray, X_valid: np.ndarray, Y_train: np.ndarray,
                 Y_valid: np.ndarray, users: int) -> (Sequential, np.ndarray, object):
    """
    Create keras model from training data and evaluate it using valid data.

    :param users:
    :param X: array with everyone's data in one block
    :param X_train: array with training data
    :param X_valid: array with valid data
    :param Y_train: array for output of training data (from model)
    :param Y_valid: array for output of valid data (from model)
    :return: model, central_vector, history data
    """
    # THIS PART NEEDS TO BE REVISED - HOW MANY NEURONS AND HOW MANY LAYERS
    model = Sequential()
    model.add(Dense(units=60, input_dim=60, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=users, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # batch size indicates the number of observations to calculate before updating the weights
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=256, batch_size=64)
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
    block_size = 60

    for person_id in eval_data.keys():
        user_blocks = []
        blocks = 0

        while (blocks + 1) * block_size < len(eval_data[person_id]):
            user_blocks.append(eval_data[person_id][blocks * block_size: (blocks + 1) * block_size])
            blocks += 1

        tst = len(user_blocks) // 5
        data_for_test = user_blocks[:tst]
        data_for_enroll = user_blocks[tst:]

        # ENROLLMENT VECTOR
        output = model.predict(np.array(data_for_enroll))
        out_vector = np.mean(output, axis=0)
        enroll[person_id] = np.subtract(out_vector, central_vector)

        # TEST VECTORS
        output = model.predict(np.array(data_for_test))
        out_vector = np.mean(output, axis=0)
        test[person_id] = np.subtract(out_vector, central_vector)

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
        a = check_score(enroll[userA], test[userA])
        confidence_TP_MLP.append(a)
        # testing with other users' test vectors
        for userB in test.keys():
            if userB != userA:
                b = check_score(enroll[userA], test[userB])
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
    X, Y, X_train, X_valid, Y_train, Y_valid, users_num = prepare_data(eval_data)
    # X_train, X_valid = feature_scaling(X_train, X_valid) <-- data preprocessing
    # Create model and central vector
    model, central_vector, history = create_model(X, X_train, X_valid, Y_train, Y_valid, users_num)
    save_model(model)
    save_central_vector(central_vector)
    # Create test users' enroll templates and test samples
    enroll, test = enroll_users(model, eval_data, central_vector)
    # Evaluate the model using cross evaluation (every user with everyone)
    conf_TP, conf_TN = cross_evaluate(enroll, test)
    # Draw figures representing confidence and save data
    confidence_figure(conf_TP, conf_TN)
    model_accuracy_figure(history)
    save_model(model)
    save_central_vector(central_vector)


def main2():
    model, central_vector = load_model_from_dir("model")
    eval_data, train_data = read_data()
    enroll, test = enroll_users(model, eval_data, central_vector)
    conf_TP, conf_TN = cross_evaluate(enroll, test)
    confidence_figure(conf_TP, conf_TN)


if __name__ == "__main__":
    # If the program is run directly (not just by importing function)
    # run the main() function
    main()
