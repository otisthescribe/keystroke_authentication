import os.path
import numpy as np
import pickle
from neural_network import load_model_from_dir, BLOCK_SIZE
from keystrokes_recorder import record


def create_template(model, samples):
    """
    Given the user's data (input) calculate the template.
    Use the model to get a vector and central vector to normalize the output.

    :param model: keras model
    :param samples: array of user's input
    :return: template vector
    """

    temp = np.array(samples)
    output = model.predict(temp)
    template = np.mean(output, axis=0)
    return template


def register_template(model):
    """
    Read user's input [probes_number] times and
    create a biometric template out of it.

    :param model: keras model
    :return: biometric template
    """

    probes_number = 10
    print(f"Sumbit your password {probes_number} times:")
    samples = []
    while len(samples) < probes_number:
        print(f"({len(samples) + 1}): ", end="")
        sample = record()
        print(sample)
        if len(sample) < BLOCK_SIZE + 1:
            print(f"\nPassword should has at least {BLOCK_SIZE + 1} characters!")
            continue
        samples.append(sample[:BLOCK_SIZE])

    template = create_template(model, samples)
    return template, samples


def user_exists(username):
    """
    Check if username already exists (cannot register again)
    """

    if os.path.exists("users_data.pickle"):
        with open("users_data.pickle", "rb") as handle:
            users = pickle.load(handle)
            if username in users.keys():
                return True


def save_user(username, template, samples):
    """
    Save user data to file.
    """

    if os.path.exists("users_data.pickle"):
        with open("users_data.pickle", "rb") as handle:
            users = pickle.load(handle)
    else:
        users = {}

    users[username] = {"template": template, "samples": samples}

    with open("users_data.pickle", "wb") as handle:
        pickle.dump(users, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    """
    Main function coordinating functions and data they return.
    """

    model, central_vector = load_model_from_dir()
    username = input("username: ")
    while user_exists(username):
        print("User already exists!")
        username = input("username: ")

    template, samples = register_template(model)
    save_user(username, template, samples)


if __name__ == "__main__":
    main()
