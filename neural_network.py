import pickle
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import keras.optimizers
from keras.layers import Dense, Input, Flatten, BatchNormalization, LSTM, Bidirectional, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from new_data.data_preprocessing import USERS, PROBE_SIZE, get_data
import sys

INPUT_SIZE = (PROBE_SIZE, 32)  # number of attributes - it will be the size of an input vector

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

    with open("./new_data/training_original.pickle", "rb") as file:
        training_original = pickle.load(file)

    with open("./new_data/testing_original.pickle", "rb") as file:
        testing_original = pickle.load(file)

    with open("./new_data/evaluation.pickle", "rb") as file:
        evaluation = pickle.load(file)

    return training_original, testing_original, evaluation


def prepare_data(train_org, test_org, evaluation):

    # print(len(train_org[0]))
    # print(len(train_org[1]))
    # print(len(test_org[0]))
    # print(len(test_org[1]))

    # exit()

    X_train = []
    Y_train = []
    X_valid = []
    Y_valid = []
    X_eval = []
    Y_eval = []

    for ind in train_org.keys():
        for sample in train_org[ind]:
            X_train.append(sample)
            if ind == 0:
                Y_train.append(0)
            else:
                Y_train.append(1)

    for ind in test_org.keys():
        for sample in test_org[ind]:
            X_valid.append(sample)
            if ind == 0:
                Y_valid.append(0)
            else:
                Y_valid.append(1)

    for ind in evaluation.keys():
        for sample in evaluation[ind]:
            X_eval.append(sample)
            if ind == 0:
                Y_eval.append(0)
            else:
                Y_eval.append(1)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)
    X_eval = np.array(X_eval)
    Y_eval = np.array(Y_eval)

    Y_oneshot = to_categorical(Y_train, num_classes=2)
    Y_oneshot2 = to_categorical(Y_valid, num_classes=2)
    Y_oneshot3 = to_categorical(Y_eval, num_classes=2)

    X_train, temp, Y_train, temp = train_test_split(X_train, Y_oneshot, test_size=1/len(X_train), random_state=123)
    X_valid, temp, Y_valid, temp = train_test_split(X_valid, Y_oneshot2, test_size=1/len(X_valid), random_state=123)
    X_eval, temp, Y_eval, temp = train_test_split(X_eval, Y_oneshot3, test_size=1/len(X_eval), random_state=123)

    return X_train, Y_train, X_valid, Y_valid, X_eval, Y_eval


def create_model(X_train, Y_train, X_valid, Y_valid, X_eval, Y_eval):
    #
    # model = Sequential()
    # model.add(Input(shape=INPUT_SIZE))
    # model.add(Flatten())
    # model.add(Dense(units=INPUT_SIZE[0]//4, activation="relu", kernel_regularizer='l2', input_shape=INPUT_SIZE))
    # model.add(Dense(units=INPUT_SIZE[0]//2, activation="relu", kernel_regularizer='l2', input_shape=INPUT_SIZE))
    # model.add(Dense(units=2, activation="sigmoid"))
    #
    # opt = keras.optimizers.Adam(learning_rate=0.001)
    # model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # model.summary()

    model = Sequential()
    model.add(Input(shape=INPUT_SIZE))
    model.add(BatchNormalization())
    forward_LSTM = LSTM(units=32, return_sequences=True)
    backward_LSTM = LSTM(units=32, return_sequences=True, go_backwards=True)
    model.add(Bidirectional(forward_LSTM, backward_layer=backward_LSTM, input_shape=INPUT_SIZE))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))

    select_optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=select_optimizer, metrics=['accuracy'])
    model.summary()

    # batch size indicates the number of observations to calculate before updating the weights
    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=64, batch_size=128)

    return model, history


def evaluate(model, X_eval, Y_eval):

    print(model.evaluate(X_eval, Y_eval, batch_size=50))

    output = model.predict(X_eval)

    print(len(output))
    TP = []
    TN = []
    for i in range(len(output)):
        if Y_eval[i][1] == 1:
            TP.append(output[i][1])

        if Y_eval[i][0] == 1:
            TN.append(output[i][1])

        print(Y_eval[i][0], ": ", output[i][0], "; ", Y_eval[i][1], ": ", output[i][1])

    print(TN)
    print(TP)

    with open("./model/confidence_TP.pickle", 'wb') as file:
        pickle.dump(TP, file)
    with open("./model/confidence_TN.pickle", 'wb') as file:
        pickle.dump(TN, file)

    return TP, TN


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
    plt.legend(["score True Positive", "score True Negative"])
    plt.xlim([-0.1, 1.1])
    plt.xlabel("score")
    plt.grid()
    plt.savefig("./plots/confidence_TP_TN.png")
    plt.show(block=False)

    # Probability of true negatives based on the threshold
    plt.figure()
    plt.plot(bins_TP[1:], np.cumsum(n_TP) / np.sum(n_TP))
    plt.plot(bins_TN[1:], 1 - (np.cumsum(n_TN) / np.sum(n_TN)))
    plt.grid()
    plt.xlabel("threshold")
    plt.ylabel("probability")
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
    plt.xlabel('threshold')
    plt.ylabel('probability')
    plt.grid()
    plt.show(block=False)
    plt.savefig("./plots/far_frr.png")

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
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.savefig("./plots/model_accuracy.png")
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

    train_org, test_org, evaluation = read_data()
    X_train, Y_train, X_valid, Y_valid, X_eval, Y_eval = prepare_data(train_org, test_org, evaluation)

    model, history = create_model(X_train, Y_train, X_valid, Y_valid, X_eval, Y_eval)
    conf_TP, conf_TN = evaluate(model, X_eval, Y_eval)

    confidence_figure(conf_TP, conf_TN)
    model_accuracy_figure(history)
    save_model(model)


if __name__ == "__main__":
    main()
