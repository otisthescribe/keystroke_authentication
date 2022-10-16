import pandas as pd
from random import randint

df = pd.DataFrame(index=range(51*400), columns=range(31 + 1))

for i in range(51):
    for j in range(400):
        df[0][i * 400 + j] = i
        for k in range(1, 32):
            df[k][i * 400 + j] = randint(0, 1000)

df.rename(columns={0: 'subject'}, inplace=True)
print(df)
df.to_csv("random_data.csv", index=False)
