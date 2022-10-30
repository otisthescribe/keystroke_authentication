import statistics
import random
import pandas as pd
import numpy as np
import os
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from progress.bar import Bar

TRAINING_USERS = 10
EVALUATION_USERS = 10
USERS = TRAINING_USERS + EVALUATION_USERS
FILES_TO_READ = 4 * USERS
PROBE_SIZE = 10
MIN_SECTIONS = 15

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


def ascii_encoding(keycodes):
    # onehot[0] -> a
    # onehot[1] -> b
    # ....
    # onehot[25] -> z
    # onehot[26] -> SPACE
    # onehot[27] -> SHIFT
    # onehot[28] -> others
    onehot = [[0] * 29] * len(keycodes)
    # ASCII A - Z => 65 -> 90
    # SPACE => 32
    # SHIFT => 16
    for i in range(len(keycodes)):
        key = keycodes[i]
        if 65 <= key <= 90:
            onehot[i][key - 65] = 1
        elif key == 32:
            onehot[i][26] = 1
        elif key == 16:
            onehot[i][27] = 1
        else:
            onehot[i][28] = 1

    df = pd.DataFrame(onehot)
    return df


def get_user_data(data):
    hold = np.array([])
    between = np.array([])
    downdown = np.array([])
    grouped = data.groupby('SECTION_ID')
    user_data = []
    for name, group in grouped:
        group.drop(['SECTION_ID', 'PARTICIPANT_ID'], inplace=True, axis=1)
        hold = np.append(hold, group['HOLD'].to_numpy())
        between = np.append(between, group['BETWEEN'].to_numpy())
        downdown = np.append(downdown, group['DOWNDOWN'].to_numpy())

        keycodes = ascii_encoding(group['KEYCODE'].to_numpy())
        group.drop(['KEYCODE'], inplace=True, axis=1)
        output = pd.concat([group, keycodes], axis=1)

        print(output)
        exit(0)

        transposed = output.T

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

    min_between = BA[int(len(BA) * (1 - 0.999))]

    sessions = data.groupby('SECTION_ID')
    bar = Bar('Remove invalid sessions', max=len(list(sessions.groups.keys())))
    for section_id, session in sessions:
        if max(session["HOLD"]) > max_hold or np.isnan(session["HOLD"].to_numpy()).any():
            data.drop(data.loc[data['SECTION_ID'] == section_id].index, inplace=True)

        elif max(session["DOWNDOWN"]) > max_downdown or np.isnan(session["BETWEEN"].to_numpy()).any():
            data.drop(data.loc[data['SECTION_ID'] == section_id].index, inplace=True)

        elif min(session["BETWEEN"]) < min_between or max(session["BETWEEN"]) > max_between or np.isnan(session["DOWNDOWN"].to_numpy()).any():
            data.drop(data.loc[data['SECTION_ID'] == section_id].index, inplace=True)

        bar.next()

    bar.finish()

    # DROP PARTICIPANTS WITH LESS THAN MIN_SECTIONS SECTIONS
    participants = data.groupby('PARTICIPANT_ID')
    user_count = len(list(participants.groups.keys()))
    bar = Bar('Remove ineligible participants', max=len(list(participants.groups.keys())))
    for participant_id, participant in participants:
        sections = participant.groupby('SECTION_ID')
        section_keys = list(sections.groups.keys())
        if len(section_keys) < MIN_SECTIONS:
            data.drop(data.loc[data['PARTICIPANT_ID'] == participant_id].index, inplace=True)
            user_count -= 1
        bar.next()

    bar.finish()

    data.reset_index(inplace=True, drop=True)
    return data, user_count


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
    bar = Bar('Loading files', max=FILES_TO_READ)
    for filename in files_list:
        if count_users < FILES_TO_READ and re.search(r"\d+_keystrokes\.txt", filename) is not None:
            try:
                file_path = "./Keystrokes/files/" + filename
                data = pd.read_csv(file_path, sep="\t", encoding="utf-8")
                data.drop(['SENTENCE', 'USER_INPUT', 'LETTERS', 'KEYSTROKE_ID'], inplace=True, axis=1)
                data.rename(columns={'TEST_SECTION_ID': 'SECTION_ID'}, inplace=True)
                data['HOLD'] = data['RELEASE_TIME'] - data['PRESS_TIME']
                data['BETWEEN'] = data['PRESS_TIME'] - data['RELEASE_TIME'].shift()
                data['DOWNDOWN'] = data['PRESS_TIME'] - data['PRESS_TIME'].shift()
                data.drop(['PRESS_TIME', 'RELEASE_TIME'], inplace=True, axis=1)

                data = data.groupby('SECTION_ID').apply(lambda group: group.iloc[1:, :])
                data_list.append(data)
                count_users += 1
            except Exception:
                continue
        else:
            break
        bar.next()

    bar.finish()

    # CREATE A DATAFRAME FROM ALL THE DATA
    new_data = pd.concat(data_list, ignore_index=True)
    hold = new_data["HOLD"].to_numpy()
    between = new_data["BETWEEN"].to_numpy()
    downdown = new_data["DOWNDOWN"].to_numpy()
    generate_figures(hold, between, downdown, suffix="raw")
    get_statistics(hold[~np.isnan(hold)], between[~np.isnan(between)], downdown[~np.isnan(downdown)], "RAW")
    # REMOVE OUTLIERS
    if REMOVE_OUTLIERS:
        new_data, user_count = remove_outliers(new_data)
        print(f"Removed {FILES_TO_READ - user_count} outliers. {user_count} users with {MIN_SECTIONS} sections left.")
        if user_count < USERS:
            print("Not enough eligible users to continue.")
            exit(0)

        hold = new_data["HOLD"].to_numpy()
        between = new_data["BETWEEN"].to_numpy()
        downdown = new_data["DOWNDOWN"].to_numpy()
        generate_figures(hold, between, downdown, suffix="outliers_removed")
        get_statistics(hold[~np.isnan(hold)], between[~np.isnan(between)], downdown[~np.isnan(downdown)], "OUTLIERS REMOVED")

    # SHUFFLE USERS AND DATA
    if DATA_SHUFFLE:
        shuffle_datastet(new_data)

    # DIVIDE USERS INTO TRAINING AND EVALUATION
    train_dataset, eval_dataset = divide_dataset(new_data)

    # STANDARDIZE THE DATA
    if DATA_AUGMENTATION:
        print("Data augmentation...")
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
    bar = Bar('Get data (training)', max=len(list(participants.groups.keys())))
    for participant_id, participant in participants:
        result, hold, between, downdown = get_user_data(participant)
        training_users[training_users_count] = result
        training_users_count += 1
        HOLD = np.append(HOLD, hold)
        BETWEEN = np.append(BETWEEN, between)
        DOWNDOWN = np.append(DOWNDOWN, downdown)
        bar.next()

    bar.finish()

    generate_figures(HOLD[~np.isnan(HOLD)], BETWEEN[~np.isnan(BETWEEN)], DOWNDOWN[~np.isnan(DOWNDOWN)], suffix="training")
    get_statistics(HOLD[~np.isnan(HOLD)], BETWEEN[~np.isnan(BETWEEN)], DOWNDOWN[~np.isnan(DOWNDOWN)], "TRAINING")

    with open("training_data.pickle", "wb") as fd:
        pickle.dump(training_users, fd)

    evaluation_users = {}

    # CREATE SAMPLES FOR EACH USER IN EVALUATION
    HOLD = np.array([])
    BETWEEN = np.array([])
    DOWNDOWN = np.array([])
    participants = eval_dataset.groupby('PARTICIPANT_ID')
    evaluation_users_count = 0
    bar = Bar('Get data (evaluation)', max=len(list(participants.groups.keys())))
    for participant_id, participant in participants:
        result, hold, between, downdown = get_user_data(participant)
        evaluation_users[evaluation_users_count] = result
        evaluation_users_count += 1
        HOLD = np.append(HOLD, hold)
        BETWEEN = np.append(BETWEEN, between)
        DOWNDOWN = np.append(DOWNDOWN, downdown)
        bar.next()

    bar.finish()

    generate_figures(HOLD[~np.isnan(HOLD)], BETWEEN[~np.isnan(BETWEEN)], DOWNDOWN[~np.isnan(DOWNDOWN)], suffix="evaluation")
    get_statistics(HOLD[~np.isnan(HOLD)], BETWEEN[~np.isnan(BETWEEN)], DOWNDOWN[~np.isnan(DOWNDOWN)], "EVALUATION")

    with open("evaluation_data.pickle", "wb") as fd:
        pickle.dump(evaluation_users, fd)


if __name__ == "__main__":
    if os.path.isdir("./Kestrokes") is False:
        print("Dowload dataset from https://userinterfaces.aalto.fi/136Mkeystrokes/ and unpack it here.")
    main()
