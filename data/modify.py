import pandas as pd
df = pd.read_csv("./data/housing.csv")
df_v1 = df.head(5000)
df_v1.to_csv("data/housing.csv", index=False)