import keyboard

keystrokes = []
key_down, key_up = 0, 0
prev_key_up, prev_key_down = 0, 0
first = True


def on_press(key):
    global key_down, key_up
    global prev_key_up, prev_key_down
    prev_key_down = key_down
    key_down = key.time


def on_release(key):
    global key_down, key_up
    global prev_key_up, prev_key_down
    global first

    prev_key_up = key_up
    key_up = key.time

    if not first:
        hold = int((key_up - key_down) * 1000)
        between = int((key_down - prev_key_up) * 1000)
        keystrokes.append(hold)
        keystrokes.append(between)
    else:
        first = False


def key_recording(key):
    if key.event_type == keyboard.KEY_DOWN:
        on_press(key)
    else:
        on_release(key)


def record():
    global keystrokes
    keystrokes = []
    keyboard.hook(key_recording)
    keyboard.wait("enter")
    keyboard.unhook(key_recording)
    return keystrokes


if __name__ == "__main__":
    print("Try it out: ", end="")
    keys = record()
    print()
    print(keys)
