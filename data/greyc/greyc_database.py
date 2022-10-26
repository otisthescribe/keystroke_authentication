import sqlite3
import pandas as pd
import numpy as np
from data_preprocessing import get_statistics, generate_figures


def change_type(df):
    for index, row in df.iterrows():
        try:
            df.iloc[index] = [int(x) for x in row]
        except Exception:
            df.drop(index, inplace=True)


conn = sqlite3.connect("keystroke.db")
cursor = conn.cursor()

print_tables = pd.read_sql_query("""SELECT name FROM sqlite_master WHERE type='table';""", conn)
print_users = pd.read_sql_query("""SELECT * FROM users""", conn)
print_data = pd.read_sql_query("""SELECT * FROM keystroke_datas""", conn)
print_typing = pd.read_sql_query("""SELECT * FROM keystroke_typing""", conn)


data = pd.DataFrame(print_data)
data.drop(['id', 'rrTime', 'vector', 'password', 'date', 'time_to_type', 'rawPress', 'rawRelease'], axis=1, inplace=True)
data = data.iloc[:, [3, 1, 0, 2]]
data.rename(columns={'prTime': 'HOLD', 'ppTime': 'DD', 'rpTime': 'BETWEEN'}, inplace=True)


data["HOLD"] = data["HOLD"].str.split(" ")
data["DD"] = data["DD"].str.split(" ")
data["BETWEEN"] = data["BETWEEN"].str.split(" ")

hold = pd.DataFrame(data["HOLD"].to_list())
dd = pd.DataFrame(data["DD"].to_list())
between = pd.DataFrame(data["BETWEEN"].to_list())

hold.drop([0, 16], axis=1, inplace=True)
dd.drop([0, 16], axis=1, inplace=True)
between.drop([0, 16], axis=1, inplace=True)

hold.columns = range(15)
dd.columns = range(15)
between.columns = range(15)

change_type(hold)
change_type(dd)
change_type(between)

HA = np.concatenate(hold.to_numpy(), axis=None)
DDA = np.concatenate(dd.to_numpy(), axis=None)
BA = np.concatenate(between.to_numpy(), axis=None)

HA = [int(x)//1000 for x in HA]
DDA = [int(x)//1000 for x in DDA]
BA = [int(x)//1000 for x in BA]

generate_figures(HA, DDA, BA, "greyc")
get_statistics(HA, DDA, BA)

new_data = pd.DataFrame(columns=range(32))
new_data[0] = data['user_id']
new_data.rename(columns={0: 'user_id'}, inplace=True)
new_data.iloc[:, 1::3] = hold
new_data.iloc[:, 2::3] = dd
new_data.iloc[:, 3::3] = between

print(new_data)

