import sqlite3
import pandas as pd

conn = sqlite3.connect("keystroke.db")
cursor = conn.cursor()

print_tables = pd.read_sql_query("""SELECT name FROM sqlite_master WHERE type='table';""", conn)
print_users = pd.read_sql_query("""SELECT * FROM users""", conn)
print_data = pd.read_sql_query("""SELECT * FROM keystroke_datas""", conn)
print_typing = pd.read_sql_query("""SELECT * FROM keystroke_typing""", conn)

# tables = pd.DataFrame(print_tables)
# users = pd.DataFrame(print_users)
data = pd.DataFrame(print_data)
data.drop(['id', 'rrTime', 'vector', 'password', 'date', 'time_to_type', 'rawPress', 'rawRelease'], axis=1, inplace=True)
data = data.iloc[:, [3, 1, 0, 2]]
# typing = pd.DataFrame(print_typing)
# typing.drop(['date', 'password', 'keyboard_number'], axis=1, inplace=True)
print(data.columns)
print(data)
