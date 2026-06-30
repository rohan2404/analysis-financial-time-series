# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


noise_std = 0.025
t = np.arange(1, 101)

rng = np.random.default_rng(12345)
a = pd.Series(rng.normal(loc=0, scale=noise_std, size=101), index=np.arange(101))
a.loc[100] = 0.01

returns = pd.DataFrame(
    {
        "a_t": a.loc[1:100].to_numpy(),
        "a_t_minus_1": a.loc[0:99].to_numpy(),
    },
    index=pd.Index(t, name="t"),
)
returns["R_t"] = returns["a_t"] + 0.2 * returns["a_t_minus_1"]

# %%
print(returns)

# %%
rolling_window = 10
returns["rolling_mean"] = returns["R_t"].rolling(window=rolling_window).mean()
returns["rolling_std"] = returns["R_t"].rolling(window=rolling_window).std()

# %%
ax = returns["R_t"].plot(title=r"$R_t$ with Rolling Mean")
returns["rolling_mean"].plot(ax=ax, label=f"{rolling_window}-period rolling mean")
ax.legend()
plt.xlabel("t")
plt.ylabel(r"$R_t$")
plt.tight_layout()

# %%
ax = returns["R_t"].plot(title=r"$R_t$ with Rolling Standard Deviation")
returns["rolling_std"].plot(ax=ax, label=f"{rolling_window}-period rolling std")
ax.legend()
plt.xlabel("t")
plt.ylabel(r"$R_t$")
plt.tight_layout()

# %%
first_half = returns["R_t"].iloc[:50]
second_half = returns["R_t"].iloc[50:]
lags = np.arange(1, 21)

acf = pd.DataFrame(
    {
        "first_half": [first_half.autocorr(lag=lag) for lag in lags],
        "second_half": [second_half.autocorr(lag=lag) for lag in lags],
    },
    index=pd.Index(lags, name="lag"),
)

print("\nAutocorrelations by half")
print(acf)

# %%
ax = acf.plot.bar(title="Autocorrelation by Lag for Each Half")
ax.axhline(0, color="black", linewidth=0.8)
plt.xlabel("t")
plt.ylabel("Autocorrelation")
plt.tight_layout()

plt.show()

# %% [markdown]
# theoretically, the ACF should be a spike at lag=1, then 0 for the rest given the
# data input we have, but t=100 is really small!
# %%

t_long = np.arange(1, 10001)

a_long = pd.Series(
    rng.normal(loc=0, scale=noise_std, size=10001),
    index=np.arange(10001),
)
a_long.loc[10000] = 0.01

returns_long = pd.DataFrame(
    {
        "a_t": a_long.loc[1:10000].to_numpy(),
        "a_t_minus_1": a_long.loc[0:9999].to_numpy(),
    },
    index=pd.Index(t_long, name="t"),
)
returns_long["R_t"] = returns_long["a_t"] + 0.2 * returns_long["a_t_minus_1"]

print("\n10,000-observation time series")
print(returns_long)

# %%
returns_long["rolling_mean"] = returns_long["R_t"].rolling(window=rolling_window).mean()
returns_long["rolling_std"] = returns_long["R_t"].rolling(window=rolling_window).std()

# %%
ax = returns_long["R_t"].plot(title=r"10,000 Observations: $R_t$ with Rolling Mean")
returns_long["rolling_mean"].plot(
    ax=ax,
    label=f"{rolling_window}-period rolling mean",
)
ax.legend()
plt.xlabel("t")
plt.ylabel(r"$R_t$")
plt.tight_layout()

# %%
ax = returns_long["R_t"].plot(
    title=r"10,000 Observations: $R_t$ with Rolling Standard Deviation"
)
returns_long["rolling_std"].plot(
    ax=ax,
    label=f"{rolling_window}-period rolling std",
)
ax.legend()
plt.xlabel("t")
plt.ylabel(r"$R_t$")
plt.tight_layout()

# %%
first_half_long = returns_long["R_t"].iloc[:5000]
second_half_long = returns_long["R_t"].iloc[5000:]

acf_long = pd.DataFrame(
    {
        "first_half": [first_half_long.autocorr(lag=lag) for lag in lags],
        "second_half": [second_half_long.autocorr(lag=lag) for lag in lags],
    },
    index=pd.Index(lags, name="lag"),
)

print("\n10,000-observation autocorrelations by half")
print(acf_long)

# %%
ax = acf_long.plot.bar(title="10,000 Observations: Autocorrelation by Lag")
ax.axhline(0, color="black", linewidth=0.8)
plt.xlabel("lag")
plt.ylabel("Autocorrelation")
plt.tight_layout()

plt.show()
# %% [markdown]
# look at above, data is clearly stationary!