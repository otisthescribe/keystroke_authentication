import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import statistics

users = {}
hold = []
between = []
downdown = []

# BAZA DANYCH Z 51 użytkownikami

data51 = pd.read_csv("DSL-StrongPasswordData.csv")
data51 = data51.drop(["sessionIndex", "rep", "subject"], axis=1)
# data51.drop(data51.iloc[:, 1::3], axis=1, inplace=True)
# data51.drop(data51.iloc[:, 0::2], axis=1, inplace=True)
data51["combined"] = data51.values.tolist()
data51.drop(data51.iloc[:, 0:-1:], axis=1, inplace=True)
user_count = -1
for index, row in data51.iterrows():
    if index // 400 > user_count:
        user_count += 1
        users[user_count] = []
    keystrokes = row["combined"]
    keystrokes = [int(x * 1000) for x in keystrokes]
    users[user_count].append(keystrokes)

    hold.append(keystrokes[::3])
    downdown.append(keystrokes[1::3])
    between.append(keystrokes[2::3])

hold = np.concatenate(np.array(hold), axis=None)
downdown = np.concatenate(np.array(downdown), axis=None)
between = np.concatenate(np.array(between), axis=None)

print(f"HOLD\n"
      f"\tMIN: {min(hold)}\n"
      f"\tMAX: {max(hold)}\n"
      f"\tMEAN: {statistics.mean(hold)}\n"
      f"\tMEDIAN: {statistics.median(hold)}\n"
      f"\tVARIANCE: {statistics.variance(hold)}")
print(f"DOWN-DOWN\n"
      f"\tMIN: {min(downdown)}\n"
      f"\tMAX: {max(downdown)}\n"
      f"\tMEAN: {statistics.mean(downdown)}\n"
      f"\tMEDIAN: {statistics.median(downdown)}\n"
      f"\tVARIANCE: {statistics.variance(downdown)}")
print(f"BETWEEN\n"
      f"\tMIN: {min(between)}\n"
      f"\tMAX: {max(between)}\n"
      f"\tMEAN: {statistics.mean(between)}\n"
      f"\tMEDIAN: {statistics.median(between)}\n"
      f"\tVARIANCE: {statistics.variance(between)}")

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
