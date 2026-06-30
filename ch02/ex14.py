# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

DATA_PATH = Path(__file__).with_name("ex14.dat")
MAX_LAG = 3
MAX_ACF_LAG = 36

data = pd.read_csv(DATA_PATH, sep=r"\s+")
# COLUMNS:
# "lnfuture": "log future price",
# "lnspot": "log spot price",
# "cost": "pct carry cost" (i.e. multiplied by 100)
data["futures_rets"] = data["lnfuture"].diff()
data["spot_rets"] = data["lnspot"].diff()
df = data[["futures_rets", "spot_rets", "cost"]]
df["cost"] /= 100
df = df.dropna()
print(df)
# %%
sns.set_theme(style="whitegrid")

axes = df.plot(
    subplots=True,
    figsize=(11, 7),
    title=["Futures returns", "Spot returns", "Carry cost"],
    legend=False,
    linewidth=1,
)
for ax in axes:
    ax.set_xlabel("Observation")
fig = axes[0].figure
fig.suptitle("Time Series", y=1.02)
fig.tight_layout()

# %%
lag_correlations = pd.DataFrame(
    {
        f"{lagged_col} lag {lag}": {
            current_col: df[current_col].corr(df[lagged_col].shift(lag))
            for current_col in df.columns
        }
        for lag in range(MAX_LAG + 1)
        for lagged_col in df.columns
    }
)

print("\nLag correlations: rows are current series, columns are lagged series")
print(lag_correlations.round(4))

fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
sns.heatmap(
    lag_correlations,
    ax=ax,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Correlation"},
)
ax.set_title("Lag Correlations")
ax.set_xlabel("Lagged series")
ax.set_ylabel("Current series")
ax.tick_params(axis="x", rotation=45)
# %% [markdown]
# Linear regression on future and spot may work well with 0.4 correlation
# There's also an abnormal correlation with spot and lag 1 through 3 on future

# %%
return_columns = ["futures_rets", "spot_rets"]
acf_lags = range(1, MAX_ACF_LAG + 1)
acf_significance_level = 1.96 / (len(df) ** 0.5)

return_acfs = pd.DataFrame(
    {
        column: [df[column].autocorr(lag=lag) for lag in acf_lags]
        for column in return_columns
    },
    index=pd.Index(acf_lags, name="lag"),
)

print("\nReturn autocorrelations")
print(return_acfs.round(4))
print(f"\nApprox. 95% ACF significance bounds: +/-{acf_significance_level:.4f}")

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
for ax, column in zip(axes, return_columns):
    ax.bar(return_acfs.index, return_acfs[column].to_numpy())
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(acf_significance_level, color="tab:red", linestyle="--", linewidth=1)
    ax.axhline(-acf_significance_level, color="tab:red", linestyle="--", linewidth=1)
    ax.fill_between(
        return_acfs.index,
        -acf_significance_level,
        acf_significance_level,
        color="tab:red",
        alpha=0.08,
    )
    ax.set_title(f"{column}: ACF")
    ax.set_ylabel("Autocorrelation")
axes[-1].set_xlabel("Lag")

# %%
adf_results = pd.DataFrame(
    {
        column: {
            "adf_statistic": result[0],
            "p_value": result[1],
            "used_lags": result[2],
            "n_observations": result[3],
            "stationary_at_5pct": result[1] < 0.05,
        }
        for column in return_columns
        for result in [adfuller(df[column], autolag="AIC")]
    }
).T

print("\nAugmented Dickey-Fuller stationarity tests")
print(adf_results.round(4))

# %%
sns.pairplot(df, diag_kind="kde", corner=True, plot_kws={"alpha": 0.6, "s": 20})
plt.show()
# %% [markdown]
# look at the spot on futures plot - it's obvious that linear regression may be the way forward
# %%
