import numpy as np
import pandas as pd
import os
import pickle

directory = "../Keystrokes - 16GB/Keystrokes/files"
user_count = 0
users = {}
set = 0
temp_count = 0

for filename in os.listdir(directory):
    if temp_count >= 1000:
        with open(f"./data/user_data_set_{set}.pickle", 'wb') as file:
            pickle.dump(users, file)
        users = {}
        temp_count = 0
        set += 1
    path = directory + "/" + filename
    try:
        print(filename)
        data = pd.read_csv(path, sep="\t", encoding="utf-8")
        # DELETE UNNECCESSARY COLUMNS AND RENAME THE REST
        data.drop(['SENTENCE', 'USER_INPUT', 'LETTER', 'KEYCODE', 'KEYSTROKE_ID'], inplace=True, axis=1)
        data.rename(columns={'TEST_SECTION_ID': 'SECTION_ID'}, inplace=True)
        data.rename(columns={'PARTICIPANT_ID': 'USER_ID'}, inplace=True)
        # SORT AND COUNT HOLD AND BETWEEN DATA
        data.sort_values(by=['USER_ID', 'RELEASE_TIME'], inplace=True, ascending=True)
        data['HOLD'] = (data['RELEASE_TIME'] - data['PRESS_TIME']) / 1000
        data['BETWEEN'] = (data['PRESS_TIME'] - data['RELEASE_TIME'].shift()) / 1000
        # GROUP AND MERGE DATA FROM EACH SESSION USER HAD
        grouped = data.groupby('SECTION_ID')  # .apply(lambda group: group.iloc[1:, 1:])
        user_data = []
        for name, group in grouped:
            hold = group['HOLD'][1:].to_numpy()
            between = group['BETWEEN'][1:].to_numpy()
            merged = [0] * (len(hold) + len(between))
            merged[::2] = hold
            merged[1::2] = between
            user_data.append(merged)
        users[user_count] = user_data
        user_count += 1
        temp_count += 1
    except Exception:
        continue

with open(f"./data/user_data_set_{set}.pickle", 'wb') as file:
    pickle.dump(users, file)
