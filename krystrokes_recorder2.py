from pynput import keyboard
import time


def count_values(keydown, keyup):
    """
    Count the keystroke values (hold and between) based on the times when
    the key was pressed and released.
    Data is represented in seconds as an integer value.
    There might be negative values as the press and release actions are not
    connected to one key.
    :param keydown: array with times of keys being pressed
    :param keyup: array with times of keys being released
    :return:
    """
    # In case we get the empty arrays
    if len(keydown) == 0 or len(keyup) == 0:
        return []

    keystrokes = []
    # Count the first value of hold and then entry the loop
    hold = int((keyup[0] - keydown[0]) * 1000)
    keystrokes.append(hold)
    for i in range(1, len(keyup)):
        # DOWN-DOWN
        downdown = int((keydown[i] - keydown[i-1]) * 1000)
        keystrokes.append(downdown)
        # BETWEEN
        between = int((keydown[i] - keyup[i-1]) * 1000)
        keystrokes.append(between)
        # HOLD
        hold = int((keyup[i] - keydown[i]) * 1000)
        keystrokes.append(hold)
    return keystrokes


def record():
    keydown = []
    keyup = []

    def on_press(key):
        nonlocal keydown
        keydown.append(time.time())

    def on_release(key):
        nonlocal keyup
        if key == keyboard.Key.enter:
            return False
        else:
            keyup.append(time.time())

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    return count_values(keydown, keyup)


if __name__ == "__main__":
    probes_number = 10
    print(f"Sumbit your password {probes_number} times:")
    samples = []
    while len(samples) < probes_number:
        print(f"({len(samples) + 1}): ", end="")
        sample = record()
        print(sample)
        samples.append(sample)

    # count = 0
    # while True:
    #     print("Try it out: ", end="")
    #     keystrokes = record()
    #     print(keystrokes)
