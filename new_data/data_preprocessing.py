import statistics
import random
import pandas as pd
import numpy as np
import os
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

USERS = 200
PROBE_SIZE = 10


def generate_figures(hold, between, downdown, suffix=""):
    plt.figure()
    plt.hist(hold, alpha=0.7, bins=50, color="orange")
    plt.xlabel("Hold time")
    plt.savefig("hold_times_" + suffix + ".png")
    plt.show(block=False)

    plt.figure()
    plt.hist(downdown, alpha=0.7, bins=50, color="green")
    plt.xlabel("Down down time")
    plt.savefig("downdown_times_" + suffix + ".png")
    plt.show(block=False)

    plt.figure()
    plt.hist(between, alpha=0.7, bins=50, color="blue")
    plt.xlabel("Between time")
    plt.savefig("between_times_" + suffix + ".png")
    plt.show(block=False)


def get_statistics(hold, between, downdown):
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


def get_user_data(data):
    hold = np.array([])
    between = np.array([])
    downdown = np.array([])
    grouped = data.groupby('SECTION_ID')
    user_data = []
    for name, group in grouped:
        group.drop(['SECTION_ID'], inplace=True, axis=1)
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

    max_hold = HA[int(len(HA) * 0.99)]
    max_downdown = DDA[int(len(DDA) * 0.99)]
    max_between = BA[int(len(BA) * 0.99)]

    sessions = data.groupby('SECTION_ID')
    for section_id, session in sessions:
        if max(session["HOLD"]) > max_hold:
            data.drop(data.loc[data['SECTION_ID'] == section_id].index, inplace=True)

        elif max(session["DOWNDOWN"]) > max_downdown:
            data.drop(data.loc[data['SECTION_ID'] == section_id].index, inplace=True)

        elif max(session["BETWEEN"]) > max_between:
            data.drop(data.loc[data['SECTION_ID'] == section_id].index, inplace=True)

    data.reset_index(inplace=True, drop=True)
    return data


def data_augmentation(data):
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
    return data


def main():
    users = {}
    directory = "./Keystrokes/files/"
    files_list = os.listdir(directory)
    random.shuffle(files_list)
    data_list = []
    count_users = 0
    for filename in files_list:
        if count_users < USERS and re.search(r"\d+_keystrokes\.txt", filename) is not None:
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
    # REMOVE OUTLIERS
    new_data = remove_outliers(new_data)
    # STANDARDIZE THE DATA
    new_data = data_augmentation(new_data)

    # CREATE SAMPLES FOR EACH USER
    HOLD = np.array([])
    BETWEEN = np.array([])
    DOWNDOWN = np.array([])
    participants = new_data.groupby('PARTICIPANT_ID')
    for participant_id, participant in participants:
        result, hold, between, downdown = get_user_data(participant)
        users[participant_id] = result
        HOLD = np.append(HOLD, hold)
        BETWEEN = np.append(BETWEEN, between)
        DOWNDOWN = np.append(DOWNDOWN, downdown)

    # GENERATE FIGURES AND STATISTICS
    generate_figures(HOLD, BETWEEN, DOWNDOWN, suffix="new_data")
    get_statistics(HOLD, BETWEEN, DOWNDOWN)

    # SAVE TO FILE
    with open("user_data.pickle", "wb") as fd:
        pickle.dump(users, fd)


if __name__ == "__main__":
    if os.path.isdir("./Kestrokes") is False:
        print("Dowload dataset from https://userinterfaces.aalto.fi/136Mkeystrokes/ and unpack it here.")
    main()
