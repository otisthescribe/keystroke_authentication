import pickle
import numpy as np
import pandas as pd
from data.data_preprocessing import generate_figures, get_statistics

hold = []
between = []
downdown = []

with open("users_data.pickle", 'rb') as file:
    users = pickle.load(file)

for user in users:
    for sample in users[user]['samples']:
        hold.append(sample[::3])
        downdown.append(sample[1::3])
        between.append(sample[2::3])

hold = np.concatenate(np.array(hold), axis=None)
downdown = np.concatenate(np.array(downdown), axis=None)
between = np.concatenate(np.array(between), axis=None)

generate_figures(hold, between, downdown)
get_statistics(hold, between, downdown)

user_dataset = pd.DataFrame()

for user in users:
    user_dataset[user] = sorted(users[user]['template'], reverse=True)

print(user_dataset)
