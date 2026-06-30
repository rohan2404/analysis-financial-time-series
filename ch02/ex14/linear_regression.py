# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).with_name("ex14.dat")

data = pd.read_csv(DATA_PATH, sep=r"\s+")
# COLUMNS:
# "lnfuture": "log future price",
# "lnspot": "log spot price",
# "cost": "pct carry cost" (i.e. multiplied by 100)
data["futures_rets"] = data["lnfuture"].diff()
data["spot_rets"] = data["lnspot"].diff()
df = data[["futures_rets", "spot_rets", "cost"]]
df = df.copy()
df["cost"] /= 100
df = df.dropna()
print(df)
# %%
