import pickle
import numpy as np
from keystrokes_recorder2 import record
from neural_network import check_score
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


def identificate_user(users, model, central_vector):
    """
    Get registered user's data (based on the login provided as input)
    and then get the second input with the rewritten sentence.
    Calculate the score and return the result.
    :param users: dictionary with users and their templates
    :param central_vector:
    :param model: keras model
    :return: biometric score (float)
    """

    print("Sumbit your password:")
    print(f"--> ", end="")
    sample = record()
    while len(sample) < BLOCK_SIZE:
        print(f"\nPassword should has at least {BLOCK_SIZE//2 + 1} characters!")
        print(f"--> ", end="")
        sample = record()
    probe = create_sample(model, sample[:BLOCK_SIZE], central_vector)

    results = {}
    for user in users:
        template = users[user]["template"]
        results[user] = check_score(template, probe)

    max_key = max(results, key=results.get)
    return max_key, results


def get_users_data():
    """
    Read user's data from file based on the username.
    :return: array with user's data
    """

    with open("user_data/users_data.pickle", "rb") as handle:
        users = pickle.load(handle)
    return users


def main():
    """
    Main function coordinating functions and data they return.
    """

    model, central_vector = load_model_from_dir()
    users = get_users_data()

    while True:
        user, results = identificate_user(users, model, central_vector)
        print(f"\nUSER: {user}\n")
        print(results)


if __name__ == "__main__":
    main()
