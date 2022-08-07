import pandas as pd

path = "../Free text dataset/"
file1 = path + "GayMarriage.csv"
file2 = path + "GunControl.csv"
file3 = path + "ReviewAMT.csv"


def transform_file(filename):
    data1 = pd.read_csv(filename, encoding="utf-8", sep="\t")
    data1.drop(['Group', 'Flow', 'Topic', 'ReviewDate', 'AccessKey', 'Opinion', 'ReviewType', 'Task', 'ReviewText'],
               inplace=True, axis=1)
    data1.rename(columns={'ReviewMeta': 'Keystrokes'}, inplace=True)
    data1.sort_values(by='UserName')
    for index, row in data1.iterrows():
        keystrokes = row['Keystrokes']
        arr = keystrokes.split(';')
        keyup = []
        keydown = []
        for e in arr:
            if 'KeyUp' in e:
                temp = e.split(' ')
                keyup.append(int(temp[0]))
            elif 'KeyDown' in e:
                temp = e.split(' ')
                keydown.append(int(temp[0]))
        keyup.sort()
        if len(keyup) < len(keydown):
            keydown = keydown[:len(keyup)]
        elif len(keyup) > len(keydown):
            keyup = keyup[:len(keydown)]
        merged = [0] * (len(keyup) + len(keydown))
        merged[::2] = keydown
        merged[1::2] = keyup
        output = []
        for i in range(3, len(merged)):
            output.append((merged[i] - merged[i-1]) / 1000)
        data1['Keystrokes'][index] = output



transform_file(file1)
transform_file(file2)
# transform_file(file3)