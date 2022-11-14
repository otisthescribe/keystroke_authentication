import pandas
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.preprocessing import StandardScaler
import random

USERS = 41  # number of users - it will be the size of an output vector


def shuffle_datastet(data):
    """
    It is recommended to shuffle the data before splitting the dataset
    into training and evaluation sets. We cannot do this using
    some external function because our data is divided into blocks of records.
    Firstly we choose two random users (blocks of 400 records) and shuffle the records
    inside them.
    Then we swap the users.
    :param data: original dataset before preprocessing
    :return: shuffled dataset
    """

    rounds = 100

    for i in range(rounds):
        index1 = random.randint(0, 50)
        index2 = random.randint(0, 50)
        # SHUFFLE THE BLOCKS
        data.iloc[index1 * 400:(index1 + 1) * 400, :].sample(frac=1)
        data.iloc[index2 * 400:(index2 + 1) * 400, :].sample(frac=1)
        # SWAP BLOCKS WITH index1 AND index2
        temp = data.iloc[index1 * 400:(index1 + 1) * 400, :].copy(deep=True)
        data.iloc[index1 * 400:(index1 + 1) * 400, :] = data.iloc[index2 * 400:(index2 + 1) * 400, :]
        data.iloc[index2 * 400:(index2 + 1) * 400, :] = temp

    return data


def read_data(filename="DSL-StrongPasswordData.csv"):
    """
    Read data from the file, split the dataset into two and
    return training and evaluation sets.
    Remove the sessionIndex and rep columns.
    :param filename: string with filename
    :return: training and evaluation datasets
    """

    data51 = pd.read_csv(filename)
    data51 = data51.drop(["sessionIndex", "rep", "H.period"], axis=1)

    # Shuffle the data before splitting
    data51 = shuffle_datastet(data51)

    train_dataset = data51.iloc[:USERS * 400, :].copy(deep=True)
    train_dataset.reset_index(inplace=True, drop=True)
    eval_dataset = data51.iloc[USERS * 400:, :].copy(deep=True)
    eval_dataset.reset_index(inplace=True, drop=True)

    del data51

    return train_dataset, eval_dataset


def data_augmentation(train_dataset, eval_dataset):
    """
    Remove the records containg outlines.
    Standardize data using StandardScaler to make
    a mean and variance exual 0.
    Perform it only on the training data.
    :param eval_dataset: evaluation DataFrame
    :param train_dataset: training DataFrame
    :return: None
    """
    DD = train_dataset.iloc[:, 1::3]
    B = train_dataset.iloc[:, 2::3]
    H = train_dataset.iloc[:, 3::3]

    HA = np.sort(np.concatenate(H.to_numpy(), axis=None))
    DDA = np.sort(np.concatenate(DD.to_numpy(), axis=None))
    BA = np.sort(np.concatenate(B.to_numpy(), axis=None))

    generate_figures(HA, BA, DDA, "raw")
    get_statistics(HA, BA, DDA)

    # Get rid of outlines

    max_hold = HA[int(len(HA) * 0.99)]
    max_downdown = DDA[int(len(DDA) * 0.99)]
    max_between = BA[int(len(BA) * 0.99)]

    # Go through the DataFrame and drop those rows that have some values greater that max

    for index, row in train_dataset.iterrows():

        if max(row[1::3]) > max_downdown:
            train_dataset.drop(index, inplace=True)

        elif max(row[2::3]) > max_between:
            train_dataset.drop(index, inplace=True)

        elif max(row[3::3]) > max_hold:
            train_dataset.drop(index, inplace=True)

    train_dataset.reset_index(inplace=True, drop=True)

    hold_scaler = StandardScaler()
    hold_scaler.fit(train_dataset.iloc[:, 3::3].values.tolist())
    downdown_scaler = StandardScaler()
    downdown_scaler.fit(train_dataset.iloc[:, 1::3].values.tolist())
    between_scaler = StandardScaler()
    between_scaler.fit(train_dataset.iloc[:, 2::3].values.tolist())

    new_data = hold_scaler.transform(train_dataset.iloc[:, 3::3].values.tolist())
    train_dataset.iloc[:, 3::3] = pandas.DataFrame(new_data)
    new_data = downdown_scaler.transform(train_dataset.iloc[:, 1::3].values.tolist())
    train_dataset.iloc[:, 1::3] = pandas.DataFrame(new_data)
    new_data = between_scaler.transform(train_dataset.iloc[:, 2::3].values.tolist())
    train_dataset.iloc[:, 2::3] = pandas.DataFrame(new_data)

    # STANDARDIZE THE EVALUATION SET WITH PARAMETERS FROM THE TRAINING SET

    for index, row in eval_dataset.iterrows():

        if max(row[1::3]) > max_downdown:
            eval_dataset.drop(index, inplace=True)

        elif max(row[2::3]) > max_between:
            eval_dataset.drop(index, inplace=True)

        elif max(row[3::3]) > max_hold:
            eval_dataset.drop(index, inplace=True)

    eval_dataset.reset_index(inplace=True, drop=True)

    new_data = hold_scaler.transform(eval_dataset.iloc[:, 3::3].values.tolist())
    eval_dataset.iloc[:, 3::3] = pandas.DataFrame(new_data)
    new_data = downdown_scaler.transform(eval_dataset.iloc[:, 1::3].values.tolist())
    eval_dataset.iloc[:, 1::3] = pandas.DataFrame(new_data)
    new_data = between_scaler.transform(eval_dataset.iloc[:, 2::3].values.tolist())
    eval_dataset.iloc[:, 2::3] = pandas.DataFrame(new_data)

    # SAVE THE SCALERS FOR FUTURE USE

    with open("../model/hold_scaler.pickle", 'wb') as file:
        pickle.dump(hold_scaler, file)

    with open("../model/downdown_scaler.pickle", 'wb') as file:
        pickle.dump(downdown_scaler, file)

    with open("../model/between_scaler.pickle", 'wb') as file:
        pickle.dump(between_scaler, file)

    return train_dataset, eval_dataset


def get_train_dict(train_dataset):
    """
    Create a dictionary out of training dataset.
    Based on subject column divide records into users and
    add to dictionary.
    Generate figures and statistics.
    :param train_dataset: training DataFrame
    :return: users dataset
    """
    users = {}
    hold, between, downdown = [], [], []
    train_dataset['combined'] = train_dataset.iloc[:, 1:].values.tolist()

    user_count = -1
    subject = "0"
    for index, row in train_dataset.iterrows():
        if row["subject"] != subject:
            subject = row["subject"]
            user_count += 1
            users[user_count] = []
        keystrokes = row["combined"]
        keystrokes = [int(x * 1000) for x in keystrokes]
        chunk = np.array(keystrokes).reshape((10, 3))
        users[user_count].append(chunk)

        hold.append(keystrokes[2::3])
        downdown.append(keystrokes[::3])
        between.append(keystrokes[1::3])

    hold = np.concatenate(np.array(hold), axis=None)
    downdown = np.concatenate(np.array(downdown), axis=None)
    between = np.concatenate(np.array(between), axis=None)

    generate_figures(hold, between, downdown, "training")
    get_statistics(hold, between, downdown)

    return users


def get_eval_dict(eval_dataset):
    """
    Create a dictionary out of evaluation dataset.
    Based on subject column divide records into users and
    add to dictionary.
    Generate figures and statistics.
    :param eval_dataset: training DataFrame
    :return: eval dataset
    """
    eval_data = {}
    hold, between, downdown = [], [], []
    eval_dataset['combined'] = eval_dataset.iloc[:, 1:].values.tolist()

    user_count = -1
    subject = "0"
    for index, row in eval_dataset.iterrows():
        if row["subject"] != subject:
            subject = row["subject"]
            user_count += 1
            eval_data[user_count] = []

        keystrokes = row["combined"]
        keystrokes = [int(x * 1000) for x in keystrokes]
        chunk = np.array(keystrokes).reshape((10, 3))
        eval_data[user_count].append(chunk)

        hold.append(keystrokes[2::3])
        downdown.append(keystrokes[::3])
        between.append(keystrokes[1::3])

    hold = np.concatenate(np.array(hold), axis=None)
    downdown = np.concatenate(np.array(downdown), axis=None)
    between = np.concatenate(np.array(between), axis=None)

    generate_figures(hold, between, downdown, "evaluation")
    get_statistics(hold, between, downdown)

    return eval_data


def generate_figures(hold, between, downdown, suffix=""):
    plt.figure()
    plt.hist(hold, alpha=0.7, bins=50, color="orange")
    plt.xlabel("Hold time " + suffix)
    plt.savefig("hold_times_" + suffix + ".png")
    plt.show(block=False)

    plt.figure()
    plt.hist(downdown, alpha=0.7, bins=50, color="green")
    plt.xlabel("Down down time " + suffix)
    plt.savefig("downdown_times_" + suffix + ".png")
    plt.show(block=False)

    plt.figure()
    plt.hist(between, alpha=0.7, bins=50, color="blue")
    plt.xlabel("Between time " + suffix)
    plt.savefig("between_times_" + suffix + ".png")
    plt.show(block=False)


def get_statistics(hold, between, downdown):
    """
    Create pandas DataFrme with statistics of three  parameters.
    :param hold: array with hold times
    :param between: array with between times
    :param downdown: array with downdowntimes
    :return: pandas DataFrame
    """
    d = {
        "MIN": [min(hold), min(between), min(downdown)],
        "MAX": [max(hold), max(between), max(downdown)],
        "MEAN": [statistics.mean(hold), statistics.mean(between), statistics.mean(downdown)],
        "MEDIAN": [statistics.median(hold), statistics.median(between), statistics.median(downdown)],
        "VARIANCE": [statistics.variance(hold), statistics.variance(between), statistics.variance(downdown)],
    }

    stat = pd.DataFrame(data=d, index=['HOLD', 'BETWEEN', 'DOWNDOWN'])
    print(stat)
    return stat


def save_data(users, eval_data):

    with open("train_user_data.pickle", "wb") as file:
        pickle.dump(users, file)

    with open("eval_user_data.pickle", "wb") as file:
        pickle.dump(eval_data, file)


def main():
    train_dataset, eval_dataset = read_data()
    train_dataset, eval_dataset = data_augmentation(train_dataset, eval_dataset)
    users = get_train_dict(train_dataset)
    eval_data = get_eval_dict(eval_dataset)
    save_data(users, eval_data)


if __name__ == "__main__":
    main()

