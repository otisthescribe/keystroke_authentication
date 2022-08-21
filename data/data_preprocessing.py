import pandas as pd
import os
import pickle

directory = "../../Keystrokes - 16GB/Keystrokes/files"
user_count = 0
train = {}
test = {}

for filename in os.listdir(directory):
    if user_count >= 400 + 200:
        with open(f"train_user_data.pickle", 'wb') as file:
            pickle.dump(train, file)
        with open(f"test_user_data.pickle", 'wb') as file:
            pickle.dump(train, file)
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
        if user_count < 400:
            train[user_count] = user_data
        else:
            test[user_count] = user_data

        user_count += 1

    except Exception:
        # Something weird happened with the data, so skip this file
        # Probably it was an encoding issue.
        continue
