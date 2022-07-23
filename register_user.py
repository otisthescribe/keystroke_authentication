import os.path  # to check if the model exists
from keystrokes_recorder import record
import numpy as np
import pickle  # to save data


def create_template(model, central_vector, data):
    block_size = 60
    i = 0
    temp = []
    while (i + 1) * block_size <= len(data):
        temp.append(data[block_size * i:block_size * (i+1)])
        i += 1
    temp = np.array(temp)
    output = model.predict(temp)
    template = np.mean(output, axis=0)
    template = np.subtract(template, central_vector)
    return template


def register_template(login):
    # Generate passphrase
    passphrase = "It was totally out of character for her."
    probes_number = 3
    print(f"Przepisz to zdanie {probes_number} razy:")
    print(passphrase)
    samples = []
    for i in range(probes_number):
        print(f"({i+1}): ", end="")
        sample = record()
        samples.append(sample)
        print()
    data = np.concatenate(np.array(samples))
    template = create_template(data)
    # print(template)
    # Sprawdzenie czy wzorzec jest poprawny
    # Wstawienie do bazy
    # Return True
    return template, passphrase


def main():
    login = input("Login: ")
    users = {}
    if os.path.exists("users_data.pickle"):
        with open("users_data.pickle", 'rb') as handle:
            users = pickle.load(handle)
        if login in users.keys():
            print("User already exists!")
            return False
    template, passphrase = register_template(login)
    # SAVE TO FILE
    users[login] = {
        "template": template,
        "passphrase": passphrase
    }
    with open("users_data.pickle", 'wb') as handle:
        pickle.dump(users, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
