import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import keras.optimizers
from keras.layers import Dense, Input, Flatten, BatchNormalization, LSTM, Bidirectional, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from new_data.data_preprocessing import TRAINING_USERS, PROBE_SIZE
import sys

INPUT_SIZE = (32, PROBE_SIZE)  # number of attributes - it will be the size of an input vector
ENROLL_SIZE = 10

np.set_printoptions(threshold=sys.maxsize)


def load_model_from_dir(directory="./model"):
    """
    Load keras model from directory passed in argument.
    If no value is passed, the default "./model" directory is used.

    :param directory: String indicating directory with neural network model
    :return: Tuple with neural network model and central vector
    """

    # Load the model, put some random data inside to initialize the structures
    model = load_model(directory)
    temp_data = [0] * 31
    model.predict(np.array([temp_data]))
    # Load the model again to already allocated structures
    model = load_model(directory)
    with open(directory + "/central_vector.pickle", "rb") as file:
        central_vector = pickle.load(file)
    return model, central_vector


def read_data():
    """
    Read data from two pickle files with training and evaluation data.

    :return: train_data and eval_data dictionaries
    """

    with open("./new_data/training_data.pickle", "rb") as file:
        train_data = pickle.load(file)

    with open("./new_data/evaluation_data.pickle", "rb") as file:
        eval_data = pickle.load(file)

    return train_data, eval_data


def check_score(sample, template):
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


def prepare_data(train_data):
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

    Y_oneshot = to_categorical(Y, num_classes=TRAINING_USERS)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_oneshot, test_size=0.2, random_state=123)
    return X, Y, X_train, X_valid, Y_train, Y_valid


def create_model(X, X_train, X_valid, Y_train, Y_valid):
    """
    Create keras model from training data and evaluate it using valid data.

    :param X: array with everyone's data in one block
    :param X_train: array with training data
    :param X_valid: array with valid data
    :param Y_train: array for output of training data (from model)
    :param Y_valid: array for output of valid data (from model)
    :return: model, central_vector, history data
    """

    # model = Sequential()
    # model.add(Input(shape=INPUT_SIZE))
    # # model.add(Dense(units=TRAINING_USERS//4, activation="relu", kernel_regularizer='l2', input_shape=INPUT_SIZE))
    # model.add(Dense(units=TRAINING_USERS//2, activation="relu", kernel_regularizer='l2', input_shape=INPUT_SIZE))
    # model.add(Flatten())
    # model.add(Dense(units=TRAINING_USERS, activation="softmax"))
    #
    # opt = keras.optimizers.Adam()
    # model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # model.summary()

    model = Sequential()
    forward_LSTM = LSTM(units=32, return_sequences=False)
    backward_LSTM = LSTM(units=32, return_sequences=False, go_backwards=True)

    model.add(Input(shape=INPUT_SIZE))
    model.add(BatchNormalization())
    model.add(Bidirectional(forward_LSTM, backward_layer=backward_LSTM, input_shape=INPUT_SIZE))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(TRAINING_USERS, activation="softmax"))

    select_optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=select_optimizer, metrics=['accuracy'])
    model.summary()

    # batch size indicates the number of observations to calculate before updating the weights
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=128, batch_size=64)
    vector_probes = model.predict(X)

    print(model.evaluate(X_valid, Y_valid, batch_size=50))

    central_vector = np.mean(vector_probes, axis=0)

    return model, central_vector, history


def enroll_users(model, eval_data, central_vector):
    """
    Create a biometric template for every user in a dictionary.

    :param model: neural network model
    :param eval_data: dictionary with user's evaluation data
    :param central_vector: array with vector from everyone's data
    :return: two dictionaries with enroll vector and test vectors for every user
    """

    enroll = {}  # key = person_id ; value = template
    test = {}  # key = person_id ; value = samples (1 or more)

    # CENTRAL VECTOR FROM EVAL DATA

    # X = []
    # for person_id in eval_data.keys():
    #     for i in eval_data[person_id]:
    #         X.append(i)
    #
    # vector_probes = model.predict(X)
    # central_vector = np.mean(vector_probes, axis=0)

    for person_id in eval_data.keys():

        # Divide the dataset into enroll and test vectors for each user
        # SEP samples for enroll and the rest for test
        SEP = ENROLL_SIZE

        # ENROLLMENT VECTOR
        temp = eval_data[person_id][:SEP]
        output = model.predict(np.array(temp))
        out_vector = np.mean(output, axis=0)
        enroll[person_id] = np.subtract(out_vector, central_vector)

        # TEST VECTORS
        test_vectors = []
        temp = eval_data[person_id][SEP:]
        output = model.predict(np.array(temp))
        for t in output:
            test_vectors.append(np.subtract(t, central_vector))
        test[person_id] = np.array(test_vectors)

    return enroll, test


def cross_evaluate(enroll, test):
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
        userA_model = enroll[userA]
        A = []
        B = []
        # testing with the same user's test vectors
        for t in test[userA]:
            a = check_score(userA_model, t)
            confidence_TP_MLP.append(a)
            A.append(a)
        # testing with other users' test vectors
        for userB in test.keys():
            if userB != userA:
                for t in test[userB]:
                    b = check_score(userA_model, t)
                    confidence_TN_MLP.append(b)
                    B.append(b)

    confidence_TP_MLP = np.squeeze(np.array(confidence_TP_MLP))
    confidence_TN_MLP = np.squeeze(np.array(confidence_TN_MLP))

    with open("./model/confidence_TP.pickle", 'wb') as file:
        pickle.dump(confidence_TP_MLP, file)
    with open("./model/confidence_TN.pickle", 'wb') as file:
        pickle.dump(confidence_TN_MLP, file)

    return confidence_TP_MLP, confidence_TN_MLP


def confidence_figure(confidence_TP_MLP, confidence_TN_MLP):
    """
    Draw two figures and save them into files for future use.

    :param confidence_TP_MLP: array with true positive attempts
    :param confidence_TN_MLP: array with true negative attempts
    """

    # Number of true negatives vs number of true positives
    plt.figure()
    n_TP, bins_TP, patches_TP = plt.hist(confidence_TP_MLP, alpha=1, bins=100)
    n_TN, bins_TN, patches_TN = plt.hist(confidence_TN_MLP, alpha=0.5, bins=100)
    plt.legend(["wyniki True Positive", "wyniki True Negative"])
    plt.xlim([-1, 1])
    plt.xlabel("wynik biometryczny")
    plt.grid()
    plt.savefig("./plots/confidence_TP_TN-50users-10.png")
    plt.show(block=False)

    # Probability of true negatives based on the threshold
    plt.figure()
    plt.plot(bins_TP[1:], np.cumsum(n_TP) / np.sum(n_TP))
    plt.plot(bins_TN[1:], 1 - (np.cumsum(n_TN) / np.sum(n_TN)))
    plt.grid()
    plt.xlabel("próg biometryczny")
    plt.ylabel("prawdopodobieństwo")
    plt.savefig("./plots/threshold_probability.png")
    plt.show(block=False)

    tn_sum = np.sum(n_TN)
    tp_sum = np.sum(n_TP)

    frr = np.cumsum(n_TP)
    for i in range(len(frr)):
        frr[i] /= tp_sum

    far = np.cumsum(n_TN)
    for i in range(len(far)):
        far[i] /= tn_sum
        far[i] = 1 - far[i]

    # FAR FRR figure
    plt.figure()
    plt.plot(bins_TP[1:], frr)
    plt.plot(bins_TN[1:], far)
    legend_f = ['false acceptance rate', 'false rejection rate']
    plt.legend(legend_f, loc='upper center')
    plt.xlabel('próg biometryczny')
    plt.ylabel('prawdopodobieństwo')
    plt.grid()
    plt.show(block=False)
    plt.savefig("./plots/far_frr-50users-10.png")

    # DET CURVE
    plt.figure(figsize=(10, 10))
    # fig, ax = plt.subplots(figsize=(10, 10))
    plt.yscale('log')
    plt.xscale('log')
    # ticks_to_use = [8, 15, 20, 27, 30, 50, 60, 70, 75, 80, 90, 100]
    # ax.set_xticks(ticks_to_use)
    # ax.set_yticks(ticks_to_use)
    plt.plot(far * 100, frr * 100)
    plt.axis([8, 100, 8, 100])
    plt.grid()
    plt.xlabel('false acceptance rate (%)')
    plt.ylabel('false rejection rate (%)')
    plt.show(block=False)
    plt.savefig("./plots/det_curve.png")


def model_accuracy_figure(history):
    """
    Draw two figures about the accuracy and loss of model during training

    :param history: historical data of model training
    """

    # Model accuracy
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Dokładność modelu")
    plt.ylabel("Dokładność")
    plt.xlabel("Epoka")
    plt.legend(["Trening", "Walidacja"], loc="upper left")
    plt.savefig("./plots/model_accuracy-50users-10.png")
    plt.show(block=False)

    # Model loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.savefig("./plots/model_loss.png")
    plt.show(block=False)


def save_model(model, directory="model"):
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


def main():
    """
    Main function to create neural network model and train it with
    test data. Then the evaluation is performed and figures are drawn.
    Model and a central vector are saved to ./model folder.
    """

    train_data, eval_data = read_data()
    X, Y, X_train, X_valid, Y_train, Y_valid = prepare_data(train_data)

    model, central_vector, history = create_model(X, X_train, X_valid, Y_train, Y_valid)
    enroll, test = enroll_users(model, eval_data, central_vector)
    conf_TP, conf_TN = cross_evaluate(enroll, test)

    confidence_figure(conf_TP, conf_TN)
    model_accuracy_figure(history)
    save_model(model)
    save_central_vector(central_vector)


if __name__ == "__main__":
    main()
