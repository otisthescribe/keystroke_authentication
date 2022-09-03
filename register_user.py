import os.path
import numpy as np
import pickle
from neural_network import load_model_from_dir, BLOCK_SIZE
from keystrokes_recorder import record
import random


def generate_passphrase():
    """
    Get the random passphrase from the file passphrase.txt.
    This file contains a lot of easy sentences that can
    be used to register and authenticate user.
    :return:
    """
    with open("./passphrase.txt") as file:
        lines = file.readlines()
        passphrase = lines[random.randint(0, len(lines) - 1)]
        passphrase = passphrase.replace("\n", "")
    return passphrase


def create_template(model, central_vector, samples):
    """
    Given the user's data (input) calculate the template.
    Use the model to get a vector and central vector to normalize the output.
    :param model: keras model
    :param central_vector: central vector saved in model directory
    :param samples: array of user's input
    :return: template vector
    """
    temp = np.array(samples)
    output = model.predict(temp)
    template = np.mean(output, axis=0)
    # template = np.subtract(template, central_vector)
    return template


def register_template(model, central_vector):
    """

    :param model:
    :param central_vector:
    :return:
    """
    probes_number = 5
    print(f"Sumbit your password {probes_number} times:")
    samples = []
    while len(samples) < probes_number:
        print(f"({len(samples) + 1}): ", end="")
        sample = record()
        if len(sample) < BLOCK_SIZE + 1:
            print(f"\nPassword should has at least {BLOCK_SIZE + 1} characters!")
            continue
        samples.append(sample[:BLOCK_SIZE])

    template = create_template(model, central_vector, samples)
    return template


def user_exists(username):
    if os.path.exists("users_data.pickle"):
        with open("users_data.pickle", "rb") as handle:
            users = pickle.load(handle)
            if username in users.keys():
                return True


def save_user(username, template):
    """
    Save user data to file.
    """
    if os.path.exists("users_data.pickle"):
        with open("users_data.pickle", "rb") as handle:
            users = pickle.load(handle)
    else:
        users = {}

    users[username] = {"template": template}

    with open("users_data.pickle", "wb") as handle:
        pickle.dump(users, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    model, central_vector = load_model_from_dir()
    username = input("username: ")
    while user_exists(username):
        print("User already exists!")
        username = input("username: ")

    template = register_template(model, central_vector)
    save_user(username, template)


if __name__ == "__main__":
    main()
