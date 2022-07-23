import pickle
import numpy as np
from keystrokes_recorder import record
from neural_network import check_score
from register_user import user_exists
from neural_network import load_model_from_dir


def create_sample(model, central_vector, data):
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


def authenticate_user(model, central_vector, username):
    user = get_user_data(username)
    template, passphrase = user["template"], user["passphrase"]
    print("Rewrite this sentence:")
    print(passphrase)
    print(f"--> ", end="")
    sample = record()
    probe = create_sample(model, central_vector, sample)
    return check_score(template, probe)


def get_user_data(username):
    with open("users_data.pickle", 'rb') as handle:
        users = pickle.load(handle)
    return users[f"{username}"]


def main():
    model, central_vector = load_model_from_dir()
    username = input("username: ")
    if not user_exists(username):
        print("User does not exist. Register users before trying to authenticate them.")
        exit(0)

    score = authenticate_user(model, central_vector, username)
    print(f"SCORE: {score * 100}%")


if __name__ == "__main__":
    main()
