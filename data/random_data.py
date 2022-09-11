import pandas as pd
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

users = {}
hold = []
between = []
downdown = []

# BAZA DANYCH Z 51 użytkownikami

user_count = -1
for i in range(400 * 51):
    if i // 400 > user_count:
        user_count += 1
        users[user_count] = []
    keystrokes = [0] * 31
    keystrokes[0:31:3] = [int(x) for x in np.random.normal(89, 40, 11)]
    keystrokes[1:31:3] = [int(x) for x in np.random.normal(248, 100, 10)]
    keystrokes[2:31:3] = [int(x) for x in np.random.normal(158, 100, 10)]
    users[user_count].append(keystrokes)

    hold.append(keystrokes[::3])
    downdown.append(keystrokes[1::3])
    between.append(keystrokes[2::3])


hold = np.concatenate(np.array(hold), axis=None)
downdown = np.concatenate(np.array(downdown), axis=None)
between = np.concatenate(np.array(between), axis=None)

plt.figure()
plt.hist(hold, alpha=0.5, bins=125, range=(0, 250), color="orange")
plt.xlabel("Hold time")
plt.savefig("./hold_times.png")
plt.show()

plt.figure()
plt.hist(downdown, alpha=0.5, bins=125, range=(0, 1000), color="orange")
plt.xlabel("Down down time")
plt.savefig("./down_down_times.png")
plt.show()

plt.hist(between, alpha=0.5, bins=125, range=(0, 1000), color="orange")
plt.xlabel("Between time")
plt.savefig("./between_times.png")
plt.show()


# CREATE TRAIN AND EVAL DICTIONARIES
# We will divide dataset into train (120 users) and eval (31 users)

eval_data = {}
eval_count = 0

for i in range(41, 51):
    eval_data[eval_count] = users.pop(i)
    eval_count += 1

# SAVE THE DATA

with open("train_user_data.pickle", "wb") as file:
    pickle.dump(users, file)

with open("eval_user_data.pickle", "wb") as file:
    pickle.dump(eval_data, file)
