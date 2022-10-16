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


class Keylogger:
    def __init__(self):
        self.keystrokes = None
        self.keydown = []
        self.keyup = []

    def on_press(self, key):
        self.keydown.append(time.time())

    def on_release(self, key):
        if key == keyboard.Key.enter:
            return False
        self.keyup.append(time.time())

    def record(self):
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        inp = input(": ")
        listener.wait()
        self.keystrokes = count_values(self.keydown, self.keyup)


if __name__ == "__main__":
    while True:
        logger = Keylogger()
        logger.record()
        print(logger.keystrokes)
        del logger

