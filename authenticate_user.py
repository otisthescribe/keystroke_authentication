import os.path
import pickle
import numpy as np
from keystrokes_recorder import record
from neural_network import check_score
from register_user import register_template


def create_sample(data):
    block_size = 60
    i = 0
    temp = []
    while (i + 1) * block_size <= len(data):
        temp.append(data[block_size * i:block_size * (i+1)])
        i += 1
    temp = np.array(temp)
    temp.reshape(i, block_size)
    output = model.predict(temp)
    sample = np.mean(output, axis=0)
    sample = np.subtract(sample, central_vector)
    return sample


def authenticate_user(login):
    if not os.path.exists("users_data.pickle"):
        register_template(login)
    with open("users_data.pickle", 'rb') as handle:
        users = pickle.load(handle)
    template, passphrase = users[login]["template"], users[login]["passphrase"]
    print("Przepisz to zdanie:")
    print(passphrase)
    print(f"--> ", end="")
    sample = record()
    probe = create_sample(sample)
    return check_score(template, probe)


def main():
    login = input("Login: ")
    score = authenticate_user(login)
    print(score)


if __name__ == "__main__":
    main()
