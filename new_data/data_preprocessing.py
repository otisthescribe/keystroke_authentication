import statistics
import random
import pandas as pd
import numpy as np
import os
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

TRAINING_USERS = 200
EVALUATION_USERS = 20
USERS = TRAINING_USERS + EVALUATION_USERS
PROBE_SIZE = 10
MIN_SECTIONS = 10

REMOVE_OUTLIERS = True
DATA_AUGMENTATION = True
DATA_SHUFFLE = False


def shuffle_datastet(data):
    for _ in range(USERS):
        participants = data.groupby('PARTICIPANT_ID')
        for participant_id, participant in participants:
            sections = participant.groupby('SECTION_ID')
            section_keys = list(sections.groups.keys())
            for _ in range(MIN_SECTIONS):
                sec1 = section_keys[random.randint(0, len(section_keys) - 1)]
                sec2 = section_keys[random.randint(0, len(section_keys) - 1)]
                temp = data.loc[data['PARTICIPANT_ID'] == sec1]
                data.loc[data['PARTICIPANT_ID'] == sec1] = data.loc[data['PARTICIPANT_ID'] == sec2]
                data.loc[data['PARTICIPANT_ID'] == sec2] = temp
        keys = list(participants.groups.keys())
        id1 = keys[random.randint(0, len(keys) - 1)]
        id2 = keys[random.randint(0, len(keys) - 1)]
        temp = data.loc[data['PARTICIPANT_ID'] == id1]
        data.loc[data['PARTICIPANT_ID'] == id1] = data.loc[data['PARTICIPANT_ID'] == id2]
        data.loc[data['PARTICIPANT_ID'] == id2] = temp

    return data


def divide_dataset(data):
    participants = data.groupby('PARTICIPANT_ID')
    keys = list(participants.groups.keys())
    if len(keys) < USERS:
        print("There is not enough users in the dataset!")
        exit(0)

    training = []
    evaluation = []
    for participants_id, participant in participants:
        if len(training) < TRAINING_USERS:
            training.append(participant)
        elif len(evaluation) < EVALUATION_USERS:
            evaluation.append(participant)
        else:
            break

    train_dataset = pd.concat(training, ignore_index=True)
    eval_dataset = pd.concat(evaluation, ignore_index=True)
    return train_dataset, eval_dataset


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


def get_statistics(hold, between, downdown, title="DATA PARAMETERS"):
    d = {
        "MIN": [min(hold), min(between), min(downdown)],
        "MAX": [max(hold), max(between), max(downdown)],
        "MEAN": [statistics.mean(hold), statistics.mean(between), statistics.mean(downdown)],
        "MEDIAN": [statistics.median(hold), statistics.median(between), statistics.median(downdown)],
        "VARIANCE": [statistics.variance(hold), statistics.variance(between), statistics.variance(downdown)],
    }
    print(f"\n{title}\n")
    stat = pd.DataFrame(data=d, index=['HOLD', 'BETWEEN', 'DOWNDOWN'])
    print(stat)
    return stat


def get_user_data(data):
    hold = np.array([])
    between = np.array([])
    downdown = np.array([])
    grouped = data.groupby('SECTION_ID')
    user_data = []
    for name, group in grouped:
        group.drop(['SECTION_ID', 'PARTICIPANT_ID'], inplace=True, axis=1)
        group.drop(index=group.index[0], axis=0, inplace=True)
        hold = np.append(hold, group['HOLD'].to_numpy())
        between = np.append(between, group['BETWEEN'].to_numpy())
        downdown = np.append(downdown, group['DOWNDOWN'].to_numpy())
        transposed = group.T
        for i in range(len(group) - PROBE_SIZE):
            chunk = transposed.iloc[:, i:i + PROBE_SIZE].to_numpy()
            user_data.append(chunk)

    return user_data, hold, between, downdown


def remove_outliers(data):
    HA = np.sort(np.concatenate(data["HOLD"].to_numpy(), axis=None))
    DDA = np.sort(np.concatenate(data["DOWNDOWN"].to_numpy(), axis=None))
    BA = np.sort(np.concatenate(data["BETWEEN"].to_numpy(), axis=None))

    max_hold = HA[int(len(HA) * 0.999)]
    max_downdown = DDA[int(len(DDA) * 0.999)]
    max_between = BA[int(len(BA) * 0.999)]

    sessions = data.groupby('SECTION_ID')
    for section_id, session in sessions:
        if max(session["HOLD"]) > max_hold:
            data.drop(data.loc[data['SECTION_ID'] == section_id].index, inplace=True)

        elif max(session["DOWNDOWN"]) > max_downdown:
            data.drop(data.loc[data['SECTION_ID'] == section_id].index, inplace=True)

        elif max(session["BETWEEN"]) > max_between:
            data.drop(data.loc[data['SECTION_ID'] == section_id].index, inplace=True)

    # DROP PARTICIPANTS WITH LESS THAN MIN_SECTIONS SECTIONS
    participants = data.groupby('PARTICIPANT_ID')
    for participant_id, participant in participants:
        sections = participant.groupby('SECTION_ID')
        section_keys = list(sections.groups.keys())
        if len(section_keys) < MIN_SECTIONS:
            data.drop(data.loc[data['PARTICIPANT_ID'] == participant_id].index, inplace=True)

    data.reset_index(inplace=True, drop=True)
    return data


def data_augmentation(data, hold_scaler=None, between_scaler=None, downdown_scaler=None):
    if hold_scaler is None or between_scaler is None or downdown_scaler is None:
        hold_scaler = StandardScaler()
        hold_scaler.fit(np.asarray(data["HOLD"]).reshape(-1, 1))
        downdown_scaler = StandardScaler()
        downdown_scaler.fit(np.asarray(data["DOWNDOWN"]).reshape(-1, 1))
        between_scaler = StandardScaler()
        between_scaler.fit(np.asarray(data["BETWEEN"]).reshape(-1, 1))

    new_data = hold_scaler.transform(np.asarray(data["HOLD"]).reshape(-1, 1))
    data["HOLD"] = pd.DataFrame(new_data)
    new_data = downdown_scaler.transform(np.asarray(data["DOWNDOWN"]).reshape(-1, 1))
    data["DOWNDOWN"] = pd.DataFrame(new_data)
    new_data = between_scaler.transform(np.asarray(data["BETWEEN"]).reshape(-1, 1))
    data["BETWEEN"] = pd.DataFrame(new_data)

    return data, hold_scaler, between_scaler, downdown_scaler


def main():
    directory = "./Keystrokes/files/"
    files_list = os.listdir(directory)
    random.shuffle(files_list)
    data_list = []
    count_users = 0
    for filename in files_list:
        if count_users < 2 * USERS and re.search(r"\d+_keystrokes\.txt", filename) is not None:
            try:
                file_path = "./Keystrokes/files/" + filename
                data = pd.read_csv(file_path, sep="\t", encoding="utf-8")
                data.drop(['SENTENCE', 'USER_INPUT', 'LETTER', 'KEYCODE', 'KEYSTROKE_ID'], inplace=True, axis=1)
                data.rename(columns={'TEST_SECTION_ID': 'SECTION_ID'}, inplace=True)
                data['HOLD'] = data['RELEASE_TIME'] - data['PRESS_TIME']
                data['BETWEEN'] = data['PRESS_TIME'] - data['RELEASE_TIME'].shift()
                data['DOWNDOWN'] = data['PRESS_TIME'] - data['PRESS_TIME'].shift()
                data.drop(['PRESS_TIME', 'RELEASE_TIME'], inplace=True, axis=1)
                data_list.append(data)
                count_users += 1
            except Exception:
                continue
        else:
            break

    # CREATE A DATAFRAME FROM ALL THE DATA
    new_data = pd.concat(data_list, ignore_index=True)
    hold = new_data["HOLD"].to_numpy()
    between = new_data["BETWEEN"].to_numpy()
    downdown = new_data["DOWNDOWN"].to_numpy()
    generate_figures(hold, between, downdown, suffix="raw")
    get_statistics(hold[~np.isnan(hold)], between[~np.isnan(between)], downdown[~np.isnan(downdown)], "RAW")
    # REMOVE OUTLIERS
    if REMOVE_OUTLIERS:
        new_data = remove_outliers(new_data)

    # SHUFFLE USERS AND DATA
    if DATA_SHUFFLE:
        shuffle_datastet(new_data)

    # DIVIDE USERS INTO TRAINING AND EVALUATION
    train_dataset, eval_dataset = divide_dataset(new_data)

    # STANDARDIZE THE DATA
    if DATA_AUGMENTATION:
        train_dataset, hold_scaler, between_scaler, downdown_scaler = data_augmentation(train_dataset)
        eval_dataset = data_augmentation(eval_dataset, hold_scaler, between_scaler, downdown_scaler)[0]
        with open("./hold_scaler.pickle", 'wb') as fd:
            pickle.dump(hold_scaler, fd)
        with open("./between_scaler.pickle", 'wb') as fd:
            pickle.dump(between_scaler, fd)
        with open("./downdown_scaler.pickle", 'wb') as fd:
            pickle.dump(downdown_scaler, fd)

    training_users = {}

    # CREATE SAMPLES FOR EACH USER IN TRAINING
    HOLD = np.array([])
    BETWEEN = np.array([])
    DOWNDOWN = np.array([])
    participants = train_dataset.groupby('PARTICIPANT_ID')
    training_users_count = 0
    for participant_id, participant in participants:
        result, hold, between, downdown = get_user_data(participant)
        training_users[training_users_count] = result
        training_users_count += 1
        HOLD = np.append(HOLD, hold)
        BETWEEN = np.append(BETWEEN, between)
        DOWNDOWN = np.append(DOWNDOWN, downdown)

    generate_figures(HOLD, BETWEEN, DOWNDOWN, suffix="training")
    get_statistics(HOLD, BETWEEN, DOWNDOWN, "TRAINING")

    with open("training_data.pickle", "wb") as fd:
        pickle.dump(training_users, fd)

    evaluation_users = {}

    # CREATE SAMPLES FOR EACH USER IN EVALUATION
    HOLD = np.array([])
    BETWEEN = np.array([])
    DOWNDOWN = np.array([])
    participants = eval_dataset.groupby('PARTICIPANT_ID')
    evaluation_users_count = 0
    for participant_id, participant in participants:
        result, hold, between, downdown = get_user_data(participant)
        evaluation_users[evaluation_users_count] = result
        evaluation_users_count += 1
        HOLD = np.append(HOLD, hold)
        BETWEEN = np.append(BETWEEN, between)
        DOWNDOWN = np.append(DOWNDOWN, downdown)

    generate_figures(HOLD, BETWEEN, DOWNDOWN, suffix="evaluation")
    get_statistics(HOLD, BETWEEN, DOWNDOWN, "EVALUATION")

    with open("evaluation_data.pickle", "wb") as fd:
        pickle.dump(evaluation_users, fd)


if __name__ == "__main__":
    if os.path.isdir("./Kestrokes") is False:
        print("Dowload dataset from https://userinterfaces.aalto.fi/136Mkeystrokes/ and unpack it here.")
    main()
