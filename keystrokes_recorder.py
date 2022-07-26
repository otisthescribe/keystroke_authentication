import keyboard


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
    """
    Record the keystrokes unless the ENTER is pressed.
    Add values of keys being pressed and released in a hooked function.
    :return: Keystrokes array
    """

    keydown = []
    keyup = []

    def key_recording(key):
        nonlocal keyup, keydown
        if key.name != 'enter':
            if key.event_type == keyboard.KEY_DOWN:
                keydown.append(key.time)
            else:
                keyup.append(key.time)

    keyboard.hook(key_recording)
    keyboard.wait("enter")
    keyboard.unhook(key_recording)
    return count_values(keydown, keyup)


if __name__ == "__main__":
    while True:
        print("Try it out: ", end="")
        keys = record()
        print(keys)
