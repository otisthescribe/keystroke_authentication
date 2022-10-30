import os.path
import numpy as np
import pandas as pd
import pickle
from neural_network import load_model_from_dir, INPUT_SIZE
from keystrokes_recorder2 import record


def data_augmentation(data):
    with open("./model/between_scaler.pickle", 'rb') as file:
        between_scaler = pickle.load(file)
    with open("./model/hold_scaler.pickle", 'rb') as file:
        hold_scaler = pickle.load(file)
    with open("./model/downdown_scaler.pickle", 'rb') as file:
        downdown_scaler = pickle.load(file)

    df = pd.DataFrame(data)

    new_data = hold_scaler.transform(df.iloc[:, 0::3].values.tolist())
    df.iloc[:, 0::3] = pd.DataFrame(new_data)
    new_data = downdown_scaler.transform(df.iloc[:, 1::3].values.tolist())
    df.iloc[:, 1::3] = pd.DataFrame(new_data)
    new_data = between_scaler.transform(df.iloc[:, 2::3].values.tolist())
    df.iloc[:, 2::3] = pd.DataFrame(new_data)

    return df.to_numpy()


def create_template(model, samples, central_vector):
    """omp
    Given the user's data (input) calculate the template.
    Use the model to get a vector and central vector to normalize the output.

    :param central_vector:
    :param model: keras model
    :param samples: array of user's input
    :return: template vector
    """

    temp = np.array(samples)
    # temp = data_augmentation(samples)
    output = model.predict(temp)
    template = np.mean(output, axis=0)
    template = np.subtract(template, central_vector)
    return template


def register_template(model, central_vector):
    """
    Read user's input [probes_number] times and
    create a biometric template out of it.

    :param central_vector:
    :param model: keras model
    :return: biometric template
    """

    probes_number = 20
    print(f"Sumbit your password {probes_number} times:")
    samples = []
    while len(samples) < probes_number:
        print(f"({len(samples) + 1}): ", end="")
        sample = record()
        print(sample)
        if len(sample) < BLOCK_SIZE:
            print(f"Password should has at least {BLOCK_SIZE // 3 + 1} characters!")
            continue
        samples.append(sample[:BLOCK_SIZE])

    template = create_template(model, samples, central_vector)
    return template, samples


def user_exists(username):
    """
    Check if username already exists (cannot register again)
    """

    if os.path.exists("user_data/users_data.pickle"):
        with open("user_data/users_data.pickle", "rb") as handle:
            users = pickle.load(handle)
            if username in users.keys():
                return True


def save_user(username, template, samples):
    """
    Save user data to file.
    """

    if os.path.exists("user_data/users_data.pickle"):
        with open("user_data/users_data.pickle", "rb") as handle:
            users = pickle.load(handle)
    else:
        users = {}

    users[username] = {"template": template, "samples": samples}

    with open("user_data/users_data.pickle", "wb") as handle:
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

    template, samples = register_template(model, central_vector)
    save_user(username, template, samples)


if __name__ == "__main__":
    main()
