import pandas as pd
import pickle

users = {}

# BAZA DANYCH Z 51 użytkownikami

data51 = pd.read_csv('DSL-StrongPasswordData.csv')
data51 = data51.drop(['sessionIndex', 'rep', 'subject'], axis=1)
data51.drop(data51.iloc[:, 0::3], axis=1, inplace=True)
data51.drop(data51.iloc[:, 0::2], axis=1, inplace=True)
data51['combined'] = data51.values.tolist()
data51.drop(data51.iloc[:, 0:-1:], axis=1, inplace=True)
user_count = -1
for index, row in data51.iterrows():
    if index // 400 > user_count:
        user_count += 1
        users[user_count] = []
    keystrokes = row['combined'][0:7]
    keystrokes = [int(x * 1000) for x in keystrokes]
    users[user_count].append(keystrokes)


# 100 users in multiple files

for i in range(1, 101):

    data = pd.read_csv("./keystrokes/keystroke100/user"+str(i)+"/latency.txt", header=None, sep="\t")
    data['combined'] = data.values.tolist()
    data.drop(data.iloc[:, 0:-1:], axis=1, inplace=True)
    users[i + 50] = []
    for index, row in data.iterrows():
        users[i + 50].append(row['combined'])

# print(len(users.keys()))

# CREATE TRAIN AND EVAL DICTIONARIES
# We will divide dataset into train (120 users) and eval (31 users)

eval_data = {}
eval_count = 0

for i in range(119, 151):
    eval_data[eval_count] = users.pop(i)
    eval_count += 1

# SAVE THE DATA

with open("train_user_data.pickle", 'wb') as file:
    pickle.dump(users, file)

with open("eval_user_data.pickle", 'wb') as file:
    pickle.dump(eval_data, file)



