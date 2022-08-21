import pickle
import numpy as np
from keystrokes_recorder import record
from neural_network import check_score
from register_user import user_exists
from neural_network import load_model_from_dir, BLOCK_SIZE


def create_sample(model, central_vector, data):
    """
    Given the user's data (input) calculate the sample.
    Use the model to get a vector and central vector to normalize the output.
    :param model: keras model
    :param central_vector: central vector saved in model directory
    :param data: user's input
    :return: sample vector
    """

    temp = np.array([data])
    output = model.predict(temp)
    sample = np.mean(output, axis=0)
    sample = np.subtract(sample, central_vector)
    return sample


def authenticate_user(model, central_vector, username):
    """
    Get registered user's data (based on the login provided as input)
    and then get the second input with the rewritten sentence.
    Calculate the score and return the result.
    :param model: keras model
    :param central_vector: central vector saved in model directory
    :param username: username provided as an input
    :return: biometric score (float)
    """
    user = get_user_data(username)
    template, passphrase = user["template"], user["passphrase"]
    print("Rewrite this sentence:")
    print(passphrase)
    print(f"--> ", end="")
    sample = record()
    while len(sample) < BLOCK_SIZE:
        print("\nRead the sentence again. Input should be at least 30 characters!")
        print(f"--> ", end="")
        sample = record()
    probe = create_sample(model, central_vector, sample[:60])
    return check_score(template, probe)


def get_user_data(username):
    """
    Read user's data from file based on the username.
    :param username: user in a users file (string)
    :return: array with user's data
    """
    with open("users_data.pickle", "rb") as handle:
        users = pickle.load(handle)
    return users[f"{username}"]


def main():
    """
    Main function coordinating functions and data they return.
    :return: None
    """
    model, central_vector = load_model_from_dir()
    username = input("username: ")
    if not user_exists(username):
        print("User does not exist. Register users before trying to authenticate them.")
        exit(0)

    score = authenticate_user(model, central_vector, username)
    print(f"SCORE: {score}")


if __name__ == "__main__":
    main()