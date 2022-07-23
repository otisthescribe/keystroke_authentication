import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense


def load_model_from_dir(directory="./model"):
    model = load_model(directory)
    with open(directory + "/central_vector.pickle", 'rb') as file:
        central_vector = pickle.load(file)
    return model, central_vector


def check_score(sample, template):
    a = np.expand_dims(sample, axis=0)
    b = np.expand_dims(template, axis=0)
    return cosine_similarity(a, b)


def read_data():
    df = pd.read_csv("sample_dataset.csv", header=None)
    train_data = {}
    eval_data = {}

    for i in range(41):
        temp = df.iloc[400 * i:400 * (i + 1)]
        train_data[i] = np.concatenate(temp.to_numpy())

    for i in range(41, 51):
        temp = df.iloc[400 * i:400 * (i + 1)]
        eval_data[i] = np.concatenate(temp.to_numpy())

    return train_data, eval_data


def prepare_data(train_data):
    X = []
    Y = []
    block_size = 60

    for person_id in train_data.keys():
        for location in range(0, 8000 - block_size, block_size):
            X.append(train_data[person_id][location:location + block_size])
            Y.append(person_id)
        # od danych każdej osoby powstaje 133 bloków po 60 cech

    X = np.array(X)
    Y = np.array(Y)

    Y_oneshot = to_categorical(Y, num_classes=41)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_oneshot, test_size=0.2, random_state=123)
    return X, Y, X_train, X_valid, Y_train, Y_valid


def create_model(X, X_train, X_valid, Y_train, Y_valid):
    model = Sequential()
    model.add(Dense(128, input_dim=60, activation='relu'))
    model.add(Dense(96, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(41, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=128, batch_size=64)
    vector_probes = model.predict(X)
    central_vector = np.mean(vector_probes, axis=0)

    return model, central_vector, history


def enroll_users(model, eval_data, central_vector):
    enroll = {}  # klucz = osoba ; wartosc = template
    test = {}  # klucz = osoba ; wartosc = tab[5]

    probes_number = 10
    for person_id in eval_data.keys():

        # ENROLLMENT VECTOR
        temp = []
        for location in range(0, probes_number):
            data = eval_data[person_id][60 * location: 60 * (location + 1)]
            temp.append(data)
        output = model.predict(np.array(temp))
        out_vector = np.mean(output, axis=0)
        enroll[person_id] = np.subtract(out_vector, central_vector)

        # TEST VECTORS
        test_vectors = []
        for i in range(probes_number):
            temp = []
            for location in range(probes_number * (i + 1), probes_number * (i + 1) + probes_number):
                data = eval_data[person_id][60 * location: 60 * (location + 1)]
                temp.append(data)
            output = model.predict(np.array(temp))
            out_vector = np.mean(output, axis=0)
            test_vectors.append(np.subtract(out_vector, central_vector))
        test[person_id] = np.array(test_vectors)

    return enroll, test


def cross_evaluate(enroll, test):
    confidence_TP_MLP = []
    confidence_TN_MLP = []
    for userA in enroll.keys():
        # utworzenie zmiennej reprezentującej model userA
        userA_model = np.expand_dims(enroll[userA], axis=0)
        # testowanie próbkami tego samego użytkownika
        for t in test[userA]:
            temp = np.expand_dims(t, axis=0)
            a = cosine_similarity(userA_model, temp)
            confidence_TP_MLP.append(a)
        # testowanie próbkami innych użytkowników
        for userB in test.keys():
            if userB != userA:
                for t in test[userB]:
                    b = cosine_similarity(userA_model, np.expand_dims(t, axis=0))
                    confidence_TN_MLP.append(b)

    confidence_TP_MLP = np.squeeze(np.array(confidence_TP_MLP))
    confidence_TN_MLP = np.squeeze(np.array(confidence_TN_MLP))

    print("TRUE POSITIVE:\n\n", confidence_TP_MLP)
    print("TRUE NEGATIVE:\n\n", confidence_TN_MLP)

    return confidence_TP_MLP, confidence_TN_MLP


def confidence_figure(confidence_TP_MLP, confidence_TN_MLP):
    plt.figure()
    n_TP, bins_TP, patches_TP = plt.hist(confidence_TP_MLP, alpha=0.5, bins=100)
    n_TN, bins_TN, patches_TN = plt.hist(confidence_TN_MLP, alpha=0.5, bins=100)
    plt.legend(['score True Positive', 'score True Negative'])
    plt.xlim([-1, 1])
    plt.xlabel('score')
    plt.grid()
    plt.show()
    plt.savefig("./plots/confidence_TP_TN.png")

    plt.figure()
    plt.plot(bins_TP[1:], np.cumsum(n_TP) / np.sum(n_TP))
    plt.plot(bins_TN[1:], 1 - (np.cumsum(n_TN) / np.sum(n_TN)))
    plt.grid()
    plt.xlabel('threshold')
    plt.ylabel('probability')
    plt.show()
    plt.savefig("./plots/threshold_probability.png")


def model_accuracy_figure(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig("./plots/model_accuracy.png")

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig("./plots/model_loss.png")


def save_model(model):
    """
    Save model to ./model folder
    :param model: keras model
    """
    model.save("model")


def save_central_vector(central_vector):
    """
    Save central vector for the future use
    :param central_vector: 1 x 41 matrix representing central vector
    """
    with open("model/central_vector.pickle", 'wb') as file:
        pickle.dump(central_vector, file, protocol=pickle.HIGHEST_PROTOCOL)


def main():
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
