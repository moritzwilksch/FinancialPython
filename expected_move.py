# %%
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10


TICKER = "CAH"
DTE = 43
P_ITM = 0.3
PERIOD = "5y"

# %%
yft = yf.Ticker(TICKER).history(PERIOD)

# %%
returns = yft["Close"].pct_change().dropna().values

samples = np.random.choice(returns, (100_000, DTE)) + 1
# args = stats.t.fit(returns)
# samples = stats.t(*args).rvs((75_000, DTE)) + 1
final_return = samples.cumprod(axis=1) - 1

lower_bound = np.quantile(final_return, P_ITM, axis=0)
upper_bound = np.quantile(final_return, 1 - P_ITM, axis=0)

# %%

N_DAYSTOPLOT = 365

if N_DAYSTOPLOT > len(yft):
    raise IndexError("Cannot plot more days than `PERIOD`")
yft_plot = yft[-N_DAYSTOPLOT:]

# sns.set_context('talk')
fig, ax = plt.subplots(figsize=(9, 5))
sns.lineplot(x=yft_plot.index[-N_DAYSTOPLOT:], y=yft_plot.Close[-N_DAYSTOPLOT:], ax=ax, color="k")
x_extrapolation = yft_plot.index.max() + np.array([np.timedelta64(i, "D") for i in range(DTE)])

ax.fill_between(
    yft_plot.index,
    yft_plot.Close,
    0.98 * yft_plot.Close.min(),
    color="0.9",
    # hatch='///',
    linewidth=0,
    edgecolor="0.5",
)

sns.despine()

fmt = "${x:,.0f}"
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)

monthyearFmt = mdates.DateFormatter("%b %y")
ax.xaxis.set_major_formatter(monthyearFmt)

ax.plot(x_extrapolation, (lower_bound + 1) * yft_plot.Close.iloc[-1], color="red")
ax.plot(x_extrapolation, (upper_bound + 1) * yft_plot.Close.iloc[-1], color="green")

ax.fill_between(
    x_extrapolation,
    (lower_bound + 1) * yft_plot.Close.iloc[-1],
    (upper_bound + 1) * yft_plot.Close.iloc[-1],
    color="white",  # '0.8',
    hatch="///",
    linewidth=0,
    edgecolor="0.5",
)


lower_final_price = (lower_bound[-1] + 1) * yft_plot.Close.iloc[-1]
upper_final_price = (upper_bound[-1] + 1) * yft_plot.Close.iloc[-1]


ax.set_title(
    f"\${TICKER} (${yft_plot.Close.iloc[-1]:.2f}) Expected Delta = {P_ITM} Move of {DTE} DTE based on prev {PERIOD}",
    size=16,
)
ax.set(
    xlim=(yft_plot.index.min(), x_extrapolation[-1] + np.timedelta64(2, "D")),
    ylim=(
        yft_plot.Close.min() * 0.98,
        1.02 * max(yft_plot.Close.max(), ((upper_bound + 1) * yft_plot.Close.iloc[-1]).max()),
    ),
)


ax.annotate(
    f"${lower_final_price:.2f}",
    xy=(x_extrapolation[-1] + np.timedelta64(2, "D"), lower_final_price),
    # size=14,
    va="center",
    color="red",
    backgroundcolor="white",
)

ax.annotate(
    f"${upper_final_price:.2f}",
    xy=(x_extrapolation[-1] + np.timedelta64(2, "D"), upper_final_price),
    # size=20,
    va="center",
    color="green",
    backgroundcolor="white",
)

plt.savefig("plot.png", transparent=False, facecolor="white", dpi=300)
