from keystrokes_recorder2 import record
from neural_network import BLOCK_SIZE
from datetime import datetime
import os
import pickle


def read_file(filename="real_data.pickle"):
    if os.path.exists(filename):
        with open(filename, "rb") as handle:
            users = pickle.load(handle)
    else:
        with open(filename, 'wb') as handle:
            users = {}
            pickle.dump(users, handle)

    return users


def get_input():
    samples = []
    print("You are starting a new password input session")
    print("If you want to quit press q when you are asked whether to keep a sample.")
    while True:
        print(f"({len(samples) + 1}): ", end="")
        sample = record()
        print(sample)
        if len(sample) < BLOCK_SIZE:
            print(f"Password should have at least {BLOCK_SIZE // 3 + 1} characters!")
            continue
        
        choice = str.lower(input("Do you want to keep that sample(y/n/q)?[y]: "))
        match choice:
            case 'n':
                continue
            case 'q':
                break
            case _:
                samples.append(sample[:BLOCK_SIZE])
    return samples


def get_session(index):
    print("You are about to start a new keystrokes input session.")
    now = datetime.now()
    date = now.strftime("%d/%m/%Y")
    time = now.strftime("%H:%M:%S")
    print(f"Today: {date}")
    print(f"Time: {time}")
    print("This is your first session" if index == 0 else f"This is your {index + 1} session. Welcome back!")
    samples = get_input()
    session = {
        "date": date,
        "time": time,
        "samples": samples
    }
    return session


def add_user(users):
    print("Think of a username and remember it!")
    username = input("username: ")
    if username in users.keys():
        print("User already exists!")
        return False

    index = 0
    get_session(index)


def update_user(users):
    pass


def list_users(users):
    pass


def main():
    users = read_file()
    menu = {
        1: "Add a new user",
        2: "Update a user",
        3: "List users",
    }
    print("Welcome to real data collector!\n")
    exit_app = False
    while exit_app is False:
        for key in sorted(menu.keys()):
            print(f"{key}) {menu[key]}")

        choice = input(f"\n"
                       f""
                       f"Select number from {min(menu.keys())} to {max(menu.keys())}: ")
        if not choice.isdigit() or int(choice) not in menu.keys():
            print("Wrong input!\n")
            continue

        match int(choice):
            case 1:
                add_user(users)
            case 2:
                update_user(users)
            case 3:
                list_users(users)


if __name__ == "__main__":
    main()
