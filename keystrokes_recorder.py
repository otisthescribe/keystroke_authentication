import keyboard

keystrokes = []
key_down, key_up = 0, 0
prev_key_up, prev_key_down = 0, 0
first = True


def on_press(key):
    """
    Function executed as the key is pressed.
    The key argument contains a lot of information as time, keycode and letter.
    :param key: keyboard object connected to the keystroke
    :return: None, global variables used
    """
    global key_down, key_up
    global prev_key_up, prev_key_down
    prev_key_down = key_down
    key_down = key.time


def on_release(key):
    """
    Function executed as the key in released.
    The key argument contains a lot of information as time, keycode and letter.
    :param key: keyboard object connected to the keystroke
    :return: None, global variables used
    """
    global key_down, key_up
    global prev_key_up, prev_key_down
    global first

    prev_key_up = key_up
    key_up = key.time

    if not first:
        # Calculate data of the keystroke (hold and between)
        hold = round(key_up - key_down, 4)
        between = round(key_down - prev_key_up, 4)
        keystrokes.append(hold)
        keystrokes.append(between)
    else:
        # Do not save the first keystroke - it cannot have the 'between' value
        first = False


def key_recording(key):
    """
    Function to determine which other function to execute
    based on the event type.
    :param key: keyboard object connected to the keystroke
    :return: None, global variables used
    """
    if key.event_type == keyboard.KEY_DOWN:
        on_press(key)
    else:
        on_release(key)


def record():
    """
    Function to coordinate the kesytroke recording.
    It hooks the key_recording function and waits for the 'enter' key.
    :return: Array of keystrokes data [hold, beteween, hold, ... , hold, between]
    """
    global keystrokes
    keystrokes = []
    keyboard.hook(key_recording)
    keyboard.wait("enter")
    keyboard.unhook(key_recording)
    return keystrokes


if __name__ == "__main__":
    keys = record()
    print(keys)
