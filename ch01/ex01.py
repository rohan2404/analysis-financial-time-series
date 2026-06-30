# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


path = Path(__file__).with_name("ex01.txt")

simple_rets = (
    pd.read_csv(path, sep=r"\s+", parse_dates=["date"], date_format="%Y%m%d")
    .set_index("date")
)

# %%
simple_rets_pct = simple_rets * 100
log_rets_pct = np.log1p(simple_rets) * 100


def return_summary(returns):
    return returns.agg(["mean", "std", "skew", "kurt", "min", "max"]).rename(
        index={"std": "standard_deviation", "kurt": "excess_kurtosis"}
    )


simple_return_summary = return_summary(simple_rets_pct)
log_return_summary = return_summary(log_rets_pct)
log_return_mean_pvalues = log_rets_pct.apply(
    lambda returns: stats.ttest_1samp(returns, popmean=0).pvalue
)

# %%
print("Simple returns, percent")
print(simple_return_summary)

print("\nLog returns, percent")
print(log_return_summary)

print("\nP-values for H0: mean log return = 0")
print(log_return_mean_pvalues)

# %%
simple_rets_pct.plot.kde(title="Density of Simple Returns")
plt.xlabel("Return (%)")
plt.tight_layout()

log_rets_pct.plot.kde(title="Density of Log Returns")
plt.xlabel("Log return (%)")
plt.tight_layout()

plt.show()
