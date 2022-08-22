import pandas as pd
import numpy as np
import os
import pickle

# CONSTANTS

LENGTH = 60  # length of an entry array in neural network
TRAIN_USERS = 400  # number of train users (it will be the length of an output vector)
EVAL_USERS = 100  # number of eval users (it will generate more data to cross evaluate)

directory = "../../Keystrokes - 16GB/Keystrokes/files"
user_count = 0
train_data = {}
eval_data = {}


for filename in os.listdir(directory):
    if user_count >= TRAIN_USERS + EVAL_USERS:
        with open(f"train_user_data.pickle", 'wb') as file:
            pickle.dump(train_data, file)
        with open(f"eval_user_data.pickle", 'wb') as file:
            pickle.dump(eval_data, file)
        print("Data preprocessing finished successfully.")
        break

    path = directory + "/" + filename
    try:
        data = pd.read_csv(path, sep="\t", encoding="utf-8")
        # DELETE UNNECCESSARY COLUMNS AND RENAME THE REST
        data.drop(['PARTICIPANT_ID', 'SENTENCE', 'USER_INPUT', 'LETTER', 'KEYCODE', 'KEYSTROKE_ID'], inplace=True, axis=1)
        data.rename(columns={'TEST_SECTION_ID': 'SECTION_ID'}, inplace=True)
        # SORT AND COUNT HOLD AND BETWEEN DATA
        # Sorting only release time - it will break the up and down times of keystrokes
        # but will prepare the data for our neural network (keylogger works the same way)
        data['RELEASE_TIME'] = data['RELEASE_TIME'].sort_values(ascending=True).values
        # data.sort_values(by=['SECTION_ID', 'RELEASE_TIME'], inplace=True, ascending=True)
        data['HOLD'] = data['RELEASE_TIME'] - data['PRESS_TIME']
        data['BETWEEN'] = data['PRESS_TIME'] - data['RELEASE_TIME'].shift()
        # GROUP AND MERGE DATA FROM EACH SESSION USER HAD
        grouped = data.groupby('SECTION_ID')  # .apply(lambda group: group.iloc[1:, 1:])
        user_data = []
        for name, group in grouped:
            hold = group['HOLD'][1:].to_numpy()
            between = group['BETWEEN'][1:].to_numpy()
            merged = [0] * (len(hold) + len(between))
            merged[::2] = hold
            merged[1::2] = between
            if len(merged) < LENGTH:
                # This sample does not fullfil the demands for our neural network
                # It does not have at least 60 elements
                raise Exception

            # Cut the data as we only need 60 elements
            merged = [int(merged[i]) for i in range(len(merged))]
            user_data.append(merged[:60])

        if user_count < TRAIN_USERS:
            train_data[user_count] = user_data
        else:
            eval_data[user_count] = user_data
        print(f"({user_count}) {filename} -- passed -- ({len(user_data)})")
        for i in user_data:
            print(f"\t{len(i)}: {i}")
        user_count += 1

    except Exception:
        # Something weird happened with the data, so skip this file
        # Probably it was an encoding issue.
        continue
