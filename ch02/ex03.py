# %%
# data loading and cleaning
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf


path = Path(__file__).with_name("ex03.txt")

df = pd.read_csv(path, sep=r"\s+")
df = df.rename(columns={
    "Year": "year",
    "Mon": "month",
    "Day": "day"
})
df["date"] = pd.to_datetime(df[df.columns[0:-1]])
df = df.set_index("date")
df = df.drop(columns=df.columns[0:-1])
print(df.tail())

# %%
df["Rate"].plot(title="Rate Time Series")
plt.xlabel("Date")
plt.ylabel("Rate")
plt.tight_layout()

# %%
lags = range(1, 37)
acf = pd.Series(
    [df["Rate"].autocorr(lag=lag) for lag in lags],
    index=pd.Index(lags, name="lag"),
    name="acf",
)

print("\nAutocorrelations")
print(acf)

# %%
plt.figure()
plt.bar(acf.index, acf.to_numpy())
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Autocorrelation Function")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.tight_layout()

# %%
transformed_rates = pd.DataFrame(
    {
        "Simple returns": df["Rate"].pct_change(),
        "Log returns": np.log(df["Rate"]).diff(),
        "Arithmetic difference": df["Rate"].diff(),
        "Second arithmetic diff": df["Rate"].diff().diff(),
    }
).dropna()


def autocorrelations(series, lags):
    return pd.Series(
        [series.autocorr(lag=lag) for lag in lags],
        index=pd.Index(lags, name="lag"),
        name="acf",
    )


def plot_series_and_acf(series, title, lags):
    acf_values = autocorrelations(series, lags)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    series.plot(ax=axes[0], title=title)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel(series.name)

    axes[1].bar(acf_values.index, acf_values.to_numpy())
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{title}: ACF")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Autocorrelation")

    plt.tight_layout()
    return acf_values


transformed_acf = {}
for column in transformed_rates:
    transformed_acf[column] = plot_series_and_acf(
        transformed_rates[column],
        column,
        lags,
    )

print("\nTransformed rate autocorrelations")
for name, acf_values in transformed_acf.items():
    print(f"\n{name}")
    print(acf_values)

# %%
rolling_window = 12
arithmetic_diff_names = ["Arithmetic difference", "Second arithmetic diff"]


def plot_rolling_diagnostics(series, title, window):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    rolling_mean.plot(ax=axes[0], color="tab:orange")
    axes[0].set_title(f"{title}: {window}-Period Rolling Mean")
    axes[0].set_ylabel("Mean")

    rolling_std.plot(ax=axes[1], color="tab:green")
    axes[1].set_title(f"{title}: {window}-Period Rolling Standard Deviation")
    axes[1].set_ylabel("Std")

    series.plot(ax=axes[2], color="tab:blue")
    axes[2].set_title(title)
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel(series.name)

    plt.tight_layout()


def split_half_acf(series, lags):
    midpoint = len(series) // 2
    first_half = series.iloc[:midpoint]
    second_half = series.iloc[midpoint:]

    return pd.DataFrame(
        {
            "first_half": autocorrelations(first_half, lags),
            "second_half": autocorrelations(second_half, lags),
        }
    )


def plot_split_half_acf(acf_values, title):
    x = np.asarray(acf_values.index)
    width = 0.4

    plt.figure()
    plt.bar(x - width / 2, acf_values["first_half"], width=width, label="First half")
    plt.bar(x + width / 2, acf_values["second_half"], width=width, label="Second half")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"{title}: Split-Half ACF")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.legend()
    plt.tight_layout()


split_half_acfs = {}
for name in arithmetic_diff_names:
    series = transformed_rates[name]
    plot_rolling_diagnostics(series, name, rolling_window)

    split_half_acfs[name] = split_half_acf(series, lags)
    plot_split_half_acf(split_half_acfs[name], name)

print("\nSplit-half autocorrelations")
for name, acf_values in split_half_acfs.items():
    print(f"\n{name}")
    print(acf_values)

# %%
second_diff = transformed_rates["Second arithmetic diff"]
pacf_lags = np.arange(1, 37)
pacf_values = pd.Series(
    pacf(second_diff, nlags=pacf_lags.max(), method="ywm")[1:],
    index=pd.Index(pacf_lags, name="lag"),
    name="pacf",
)
pacf_confidence = 1.96 / np.sqrt(len(second_diff))

print("\nSecond arithmetic diff PACF")
print(pacf_values)

# %%
plt.figure()
plt.bar(pacf_values.index, pacf_values.to_numpy())
plt.axhline(0, color="black", linewidth=0.8)
plt.axhline(pacf_confidence, color="red", linestyle="--", linewidth=0.8)
plt.axhline(-pacf_confidence, color="red", linestyle="--", linewidth=0.8)
plt.title("Second Arithmetic Diff: PACF")
plt.xlabel("Lag")
plt.ylabel("Partial autocorrelation")
plt.tight_layout()

# %%
max_ar_order = 8
max_ma_order = 8


def ar_residuals(series, order):
    values = pd.Series(series.to_numpy())

    if order == 0:
        return values - values.mean()

    model = AutoReg(values, lags=order, old_names=False)
    return model.fit().resid


eacf_values = pd.DataFrame(
    index=pd.Index(range(max_ar_order + 1), name="AR order"),
    columns=pd.Index(range(max_ma_order + 1), name="MA order"),
    dtype=float,
)

eacf_significant = pd.DataFrame(
    False,
    index=eacf_values.index,
    columns=eacf_values.columns,
)
for ar_order in eacf_values.index:
    residuals = ar_residuals(second_diff, ar_order)

    for ma_order in eacf_values.columns:
        residual_acf = residuals.autocorr(lag=ma_order + 1)
        threshold = 1.96 / np.sqrt(len(residuals))

        eacf_values.loc[ar_order, ma_order] = residual_acf
        eacf_significant.loc[ar_order, ma_order] = abs(residual_acf) > threshold

eacf_symbols = pd.DataFrame(
    np.where(eacf_significant, "x", "o"),
    index=eacf_significant.index,
    columns=eacf_significant.columns,
)

print("\nSecond arithmetic diff EACF values")
print(eacf_values)

print("\nSecond arithmetic diff EACF symbols (o = insignificant, x = significant)")
print(eacf_symbols)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(eacf_significant.astype(int), cmap="Greys", aspect="auto")

ax.set_title("Second Arithmetic Diff: EACF")
ax.set_xlabel("MA order")
ax.set_ylabel("AR order")
ax.set_xticks(range(max_ma_order + 1))
ax.set_yticks(range(max_ar_order + 1))

for ar_order in eacf_values.index:
    for ma_order in eacf_values.columns:
        ax.text(
            ma_order,
            ar_order,
            eacf_symbols.loc[ar_order, ma_order],
            ha="center",
            va="center",
            color="tab:red" if eacf_significant.loc[ar_order, ma_order] else "black",
        )

plt.tight_layout()

# %%
rate = df["Rate"].asfreq("MS")
model_order = (1, 1, 2)
model_label = f"ARIMA{model_order}"
arima_model = ARIMA(rate, order=model_order, trend="n")
arima_fit = arima_model.fit()

forecast_result = arima_fit.get_forecast(steps=4)
forecast_summary = forecast_result.summary_frame(alpha=0.05)
forecast_summary.index = pd.date_range("2009-04-01", periods=4, freq="MS")
forecast_summary = forecast_summary.rename(
    columns={
        "mean": "forecast",
        "mean_se": "standard_error",
        "mean_ci_lower": "lower_95",
        "mean_ci_upper": "upper_95",
    }
)

residuals = arima_fit.resid.dropna()
fit_metrics = pd.Series(
    {
        "aic": arima_fit.aic,
        "bic": arima_fit.bic,
        "hqic": arima_fit.hqic,
        "log_likelihood": arima_fit.llf,
        "residual_mean": residuals.mean(),
        "residual_std": residuals.std(),
        "residual_mae": residuals.abs().mean(),
        "residual_rmse": np.sqrt((residuals ** 2).mean()),
    },
    name="fit_metrics",
)

ljung_box = acorr_ljungbox(residuals, lags=[12, 24, 36], return_df=True)

print(f"\n{model_label} fit summary")
print(arima_fit.summary())

print(f"\n{model_label} fit metrics")
print(fit_metrics)

print("\nResidual Ljung-Box tests")
print(ljung_box)

print("\nForecasts for April-July 2009")
print(forecast_summary)

# %%
forecast_plot_start = "2006-01-01"
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
rate.loc[forecast_plot_start:].plot(ax=ax, label="Observed")
forecast_summary["forecast"].plot(ax=ax, label="Forecast", color="tab:orange")
ax.fill_between(
    forecast_summary.index,
    forecast_summary["lower_95"],
    forecast_summary["upper_95"],
    color="tab:orange",
    alpha=0.2,
    label="95% forecast interval",
)
ax.set_title(f"{model_label} Forecasts for Rate")
ax.set_xlabel("Date")
ax.set_ylabel("Rate")
ax.legend()

# %%
residual_acf = autocorrelations(residuals, lags)
residual_moments = pd.Series(
    {
        "mean": residuals.mean(),
        "variance": residuals.var(),
        "skewness": stats.skew(residuals),
    },
    name="residual_moments",
)

white_noise_test = acorr_ljungbox(residuals, lags=[12, 24, 36], return_df=True)
white_noise_test["reject_white_noise_5pct"] = white_noise_test["lb_pvalue"] < 0.05

print(f"\n{model_label} raw residual moments")
print(residual_moments)
print(
    "\nRaw residual kurtosis is omitted because ARIMA residuals can include "
    "initialization effects from the differencing/state-space filter. A few early "
    "raw residuals can dominate kurtosis, so the SARIMAX summary's ordinary "
    "kurtosis on standardized forecast errors is the better tail diagnostic."
)

print("\nWhite-noise hypothesis test at 5% level")
print(white_noise_test)

if white_noise_test["reject_white_noise_5pct"].any():
    print("\nConclusion: reject white noise at the 5% level for at least one lag.")
else:
    print("\nConclusion: fail to reject white noise at the 5% level.")

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)

residuals.plot(ax=axes[0])
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title(f"{model_label} Residuals")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Residual")

residuals.plot.kde(ax=axes[1])
axes[1].set_title(f"{model_label} Residual Density")
axes[1].set_xlabel("Residual")

axes[2].bar(residual_acf.index, residual_acf.to_numpy())
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_title(f"{model_label} Residual ACF")
axes[2].set_xlabel("Lag")
axes[2].set_ylabel("Autocorrelation")

# %%
def model_diagnostics_row(name, fit, residual_lag=24):
    model_residuals = fit.resid.dropna()
    lb_test = acorr_ljungbox(model_residuals, lags=[residual_lag], return_df=True)
    mle_retvals = getattr(fit, "mle_retvals", {})

    return {
        "model": name,
        "converged": mle_retvals.get("converged", np.nan),
        "aic": fit.aic,
        "bic": fit.bic,
        "hqic": fit.hqic,
        "log_likelihood": fit.llf,
        "residual_rmse": np.sqrt((model_residuals ** 2).mean()),
        f"ljung_box_pvalue_lag_{residual_lag}": lb_test["lb_pvalue"].iloc[0],
    }


def fit_arima_candidate(series, name, order, seasonal_order=(0, 0, 0, 0)):
    try:
        candidate_model = ARIMA(
            series,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        candidate_fit = candidate_model.fit(method_kwargs={"maxiter": 300})
        return model_diagnostics_row(name, candidate_fit)
    except Exception as error:
        return {
            "model": name,
            "converged": False,
            "aic": np.nan,
            "bic": np.nan,
            "hqic": np.nan,
            "log_likelihood": np.nan,
            "residual_rmse": np.nan,
            "ljung_box_pvalue_lag_24": np.nan,
            "error": str(error),
        }


nearby_arima_results = []
for p in range(0, 3):
    for q in range(0, 4):
        nearby_arima_results.append(
            fit_arima_candidate(
                rate,
                name=f"ARIMA({p},1,{q})",
                order=(p, 1, q),
            )
        )

nearby_arima_comparison = (
    pd.DataFrame(nearby_arima_results)
    .sort_values("aic", na_position="last")
    .reset_index(drop=True)
)

print("\nNearby ARIMA(p,1,q) model comparison")
print(nearby_arima_comparison)

# %%
seasonal_candidates = [
    {"order": (1, 1, 2), "seasonal_order": (1, 0, 0, 12)},
    {"order": (1, 1, 2), "seasonal_order": (0, 0, 1, 12)},
    {"order": (1, 1, 2), "seasonal_order": (1, 0, 1, 12)},
    {"order": (1, 1, 2), "seasonal_order": (0, 1, 1, 12)},
]

seasonal_results = []
for candidate in seasonal_candidates:
    seasonal_results.append(
        fit_arima_candidate(
            rate,
            name=f"ARIMA{candidate['order']}x{candidate['seasonal_order']}",
            order=candidate["order"],
            seasonal_order=candidate["seasonal_order"],
        )
    )

seasonal_comparison = (
    pd.DataFrame(seasonal_results)
    .sort_values("aic", na_position="last")
    .reset_index(drop=True)
)

print("\nSimple seasonal model comparison, s=12")
print(seasonal_comparison)

# %%
def fractional_difference(series, d, threshold=1e-4):
    weights = [1.0]
    k = 1

    while True:
        next_weight = -weights[-1] * (d - k + 1) / k
        if abs(next_weight) < threshold:
            break
        weights.append(next_weight)
        k += 1

    weights = np.asarray(weights)
    values = series.to_numpy()
    differenced = np.full(len(values), np.nan)

    for idx in range(len(weights) - 1, len(values)):
        window = values[idx - len(weights) + 1 : idx + 1]
        differenced[idx] = np.dot(weights, window[::-1])

    return pd.Series(differenced, index=series.index, name=f"frac_diff_d_{d:.2f}").dropna()


fractional_results = []
fractional_d_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for d in fractional_d_values:
    frac_diff_rate = fractional_difference(rate, d=d)

    for p in range(0, 3):
        for q in range(0, 4):
            fractional_results.append(
                fit_arima_candidate(
                    frac_diff_rate,
                    name=f"FracDiff(d={d:.1f}) + ARMA({p},{q})",
                    order=(p, 0, q),
                )
            )

fractional_comparison = (
    pd.DataFrame(fractional_results)
    .sort_values("aic", na_position="last")
    .reset_index(drop=True)
)

print("\nFractional differencing ARMA comparison")
print(fractional_comparison.head(20))

# %%
comparison_sets = {
    "Nearby ARIMA": nearby_arima_comparison,
    "Seasonal SARIMA": seasonal_comparison,
    "Fractional differencing": fractional_comparison.head(12),
}

for title, comparison in comparison_sets.items():
    plt.figure(figsize=(10, 5))
    plt.bar(comparison["model"], comparison["aic"])
    plt.title(f"{title}: AIC Comparison")
    plt.xlabel("Model")
    plt.ylabel("AIC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

plt.show()
# %%
