# %%
from typing import Literal
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from typing import Literal
from joblib import delayed, Parallel
import yfinance as yf

plt.rcParams["font.family"] = "Arial"


TICKER = "CAH"
DTE = 53
P_ITM = 0.3
PERIOD = "5y"

# %%
yft = yf.Ticker(TICKER).history(PERIOD)

# %%
returns = yft["Close"].pct_change().fillna(0).values


# %%
THRESH = "lower"  # lower, upper, both

start = len(returns) // 5


def simulate(
    cutoff, n_samples=2_500, thresh: Literal["lower", "upper", "both"] = "both"
):
    samples = np.random.choice(returns[:cutoff], (n_samples, DTE)) + 1
    final_return = samples.cumprod(axis=1) - 1

    lower_bound = np.quantile(final_return[:, -1], P_ITM)
    upper_bound = np.quantile(final_return[:, -1], 1 - P_ITM)

    lower = (lower_bound + 1) * yft["Close"].iloc[cutoff]
    upper = (upper_bound + 1) * yft["Close"].iloc[cutoff]
    return (
        lower < yft["Close"].iloc[cutoff + DTE] < upper
        if thresh == "both"
        else yft["Close"].iloc[cutoff + DTE] < upper
        if thresh == "upper"
        else yft["Close"].iloc[cutoff + DTE] > lower
    )


result = Parallel(n_jobs=-1)(
    delayed(simulate)(cutoff, thresh=THRESH)
    for cutoff in range(start, len(returns) - DTE)
)

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(
    list(range(start, len(returns) - DTE)),
    (np.array(result).cumsum() / range(1, len(result) + 1)),
    c="blue",
    zorder=20,
)

theoretical_p = (
    1 - (2 * P_ITM) if THRESH == "both" else 1 - P_ITM
)  # if THRESH == 'lower' else P_ITM
ax.axhline(theoretical_p, ls="--", color="k", zorder=5, alpha=0.5)
labeldict = {
    "lower": "$price > lower$",
    "upper": "$price < upper$",
    "both": "$lower < price < upper$",
}
ax.set(
    # xticks=range(start, len(returns) - DTE),
    xlabel="Days used for empirical sampling",
    ylabel=f"Empirical $\mathbf{{{'P'}}}$({labeldict[THRESH]})",
)
ax.text(
    len(returns - DTE - start) + 40,
    y=theoretical_p,
    s=f"Theoretical\n$\mathbf{{{'P'}}}$({labeldict[THRESH]})",
    ha="left",
    va="center",
)
sns.despine()

# %%
