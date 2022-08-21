import numpy as np
import pandas as pd
import pickle


def transform_file(file, drop_columns):
    data = pd.read_csv(file, encoding="utf-8", sep="\t")
    data.drop(drop_columns, inplace=True, axis=1)
    data.rename(columns={"ReviewMeta": "Keystrokes"}, inplace=True)
    data.sort_values(by="UserName", inplace=True)

    for index, row in data.iterrows():
        keystrokes = row["Keystrokes"]
        arr = keystrokes.split(";")
        keyup = []
        keydown = []
        for i, e in enumerate(arr):
            if "KeyUp" in e:
                temp = e.split(" ")
                keyup.append(int(temp[0]))
            elif "KeyDown" in e:
                temp = e.split(" ")
                keydown.append(int(temp[0]))

        # keyup.sort()
        if len(keyup) < len(keydown):
            keydown = keydown[: len(keyup)]
        elif len(keyup) > len(keydown):
            keyup = keyup[: len(keydown)]
        merged = [0] * (len(keyup) + len(keydown))
        merged[::2] = keydown
        merged[1::2] = keyup
        output = []
        for i in range(3, len(merged)):
            output.append((merged[i] - merged[i - 1]) / 1000)
        data["Keystrokes"][index] = output

    # Save the data into CSV file with _01.csv suffix
    filename = file[:-4] + "_01.csv"
    data.to_csv(filename, header=True, index=False)
    # Aggregate data from one user into one row in the output dataset
    # Each user has 4 rows in the starting dataset
    new_data = data.groupby("UserName", as_index=False).agg({"Keystrokes": "sum"})
    new_data.sort_values(by="UserName", inplace=True)
    # Save the data into CSV file with _01.csv suffix
    filename = file[:-4] + "_02.csv"
    new_data.to_csv(filename, header=True, index=False)

    # TEST
    print(data['Keystrokes'][0])
    print(data['Keystrokes'][1])
    print(data['Keystrokes'][2])
    print(data['Keystrokes'][3])
    print(new_data['Keystrokes'][0])

    return new_data


def merge():
    file1 = "GayMarriage.csv"
    file2 = "GunControl.csv"
    file3 = "ReviewAMT.csv"

    drop_columns = [
        "Group",
        "Flow",
        "Topic",
        "ReviewDate",
        "AccessKey",
        "Opinion",
        "ReviewType",
        "Task",
        "ReviewText",
    ]
    data1 = transform_file(file1, drop_columns)
    data2 = transform_file(file2, drop_columns)
    drop_columns = [
        "Group",
        "Flow",
        "Restaurant",
        "ReviewDate",
        "AccessKey",
        "Addr",
        "ReviewTopic",
        "Site",
        "Task",
        "ReviewText",
    ]
    data3 = transform_file(file3, drop_columns)

    joined = pd.concat([data1, data2, data3], ignore_index=True)
    joined["UserName"] = joined["UserName"].str.upper()
    output = joined.groupby("UserName", as_index=False).agg({"Keystrokes": "sum"})
    output["LENGTH"] = output["Keystrokes"].apply(lambda x: len(x))
    output["LENGTH"] = output["Keystrokes"].apply(lambda x: len(x))
    output.sort_values(by=["LENGTH"], inplace=True, ascending=False, ignore_index=True)
    return output


def create_dict(data):
    users = {}
    for index, row in data.iterrows():
        users[index] = np.array(row["Keystrokes"])
    save_to_file(users)


def save_to_file(dictionary):
    with open("free_text_data.pickle", "wb") as file:
        pickle.dump(dictionary, file)


if __name__ == "__main__":
    data = merge()
    create_dict(data)
