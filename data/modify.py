from pathlib import Path

import pandas as pd


data_file = Path(__file__).with_name("housing.csv")
df = pd.read_csv(data_file)
df_v1 = df.head(5000)
df_v1.to_csv(data_file, index=False)
