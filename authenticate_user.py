import pickle
import numpy as np
from keystrokes_recorder2 import record
from neural_network import check_score
from register_user import user_exists
from neural_network import load_model_from_dir, BLOCK_SIZE


def create_sample(model, data, central_vector):
    """
    Given the user's data (input) calculate the sample.
    Use the model to get a vector and central vector to normalize the output.
    :param central_vector:
    :param model: keras model
    :param data: user's input
    :return: sample vector
    """

    temp = np.array([data])
    output = model.predict(temp)
    sample = np.mean(output, axis=0)
    sample = np.subtract(sample, central_vector)
    return sample


def authenticate_user(model, username, central_vector):
    """
    Get registered user's data (based on the login provided as input)
    and then get the second input with the rewritten sentence.
    Calculate the score and return the result.
    :param central_vector:
    :param model: keras model
    :param username: username provided as an input
    :return: biometric score (float)
    """

    user = get_user_data(username)
    template = user["template"]
    print("Sumbit your password:")
    print(f"--> ", end="")
    sample = record()
    while len(sample) < BLOCK_SIZE:
        print(f"Password should has at least {BLOCK_SIZE//2 + 1} characters!")
        print(f"--> ", end="")
        sample = record()
    probe = create_sample(model, sample[:BLOCK_SIZE], central_vector)
    return check_score(template, probe)


def get_user_data(username):
    """
    Read user's data from file based on the username.
    :param username: user in a users file (string)
    :return: array with user's data
    """

    with open("user_data/users_data.pickle", "rb") as handle:
        users = pickle.load(handle)
    return users[f"{username}"]


def main():
    """
    Main function coordinating functions and data they return.
    """

    model, central_vector = load_model_from_dir()
    username = input("username: ")
    if not user_exists(username):
        print("User does not exist. Register users before trying to authenticate them.")
        exit(0)

    while True:
        score = authenticate_user(model, username, central_vector)
        print(f"SCORE: {score}")


if __name__ == "__main__":
    main()
