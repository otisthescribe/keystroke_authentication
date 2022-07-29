import os.path
import numpy as np
import pickle
import diceware
from neural_network import load_model_from_dir
from keystrokes_recorder import record
import random


def generate_passphrase2():
    num = 4
    options = diceware.handle_options(args=[f"-n {num}", "-d "])
    passphrase = diceware.get_passphrase(options=options)
    return passphrase


def generate_passphrase():
    with open("./pass.txt") as file:
        lines = file.readlines()
        passphrase = lines[random.randint(0, len(lines) - 1)]
        passphrase = passphrase.replace("\n", "")
    return passphrase


def create_template(model, central_vector, data):
    block_size = 60
    i = 0
    temp = []
    while (i + 1) * block_size <= len(data):
        temp.append(data[block_size * i : block_size * (i + 1)])
        i += 1
    temp = np.array(temp)
    output = model.predict(temp)
    template = np.mean(output, axis=0)
    template = np.subtract(template, central_vector)
    return template


def register_template(model, central_vector):
    passphrase = generate_passphrase()
    probes_number = 5
    print(f"Rewrite this sentence {probes_number} times:")
    print(passphrase)
    samples = []
    for i in range(probes_number):
        print(f"({i+1}): ", end="")
        sample = record()
        samples.append(sample)
    data = np.concatenate(np.array(samples))
    template = create_template(model, central_vector, data)
    return template, passphrase


def user_exists(username):
    if os.path.exists("users_data.pickle"):
        with open("users_data.pickle", "rb") as handle:
            users = pickle.load(handle)
            if username in users.keys():
                return True


def save_user(username, template, passphrase):
    """
    Save user data to file.
    """
    if os.path.exists("users_data.pickle"):
        with open("users_data.pickle", "rb") as handle:
            users = pickle.load(handle)
    else:
        users = {}

    users[username] = {"template": template, "passphrase": passphrase}

    with open("users_data.pickle", "wb") as handle:
        pickle.dump(users, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    model, central_vector = load_model_from_dir()
    username = input("username: ")
    while user_exists(username):
        print("User already exists!")
        username = input("username: ")

    template, passphrase = register_template(model, central_vector)
    save_user(username, template, passphrase)


if __name__ == "__main__":
    main()
