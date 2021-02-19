# %%
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats.stats import ks_1samp
import seaborn as sns
import yfinance as yf

TICKER = "T"
DTE = 28
P_ITM = 0.2
PERIOD = "6mo"

# %%
yft = yf.Ticker(TICKER).history(PERIOD)

# %%
returns = yft['Close'].pct_change().dropna().values

samples = np.random.choice(returns, (50_000, DTE)) + 1
final_return = samples.cumprod(axis=1) - 1

# %%
lower_bound = np.quantile(final_return, P_ITM, axis=0)
upper_bound = np.quantile(final_return, 1-P_ITM, axis=0)

# plt.plot(lower_bound)
# plt.plot(upper_bound)

# %%
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(15, 8))
sns.lineplot(yft.index, yft.Close, ax=ax, color='k')
x_extrapolation = yft.index.max() + np.array([np.timedelta64(i, 'D') for i in range(DTE)])

ax.fill_between(
    yft.index,
    yft.Close,
    0.98 * yft.Close.min(),
    color='0.9',
    #hatch='///',
    linewidth=0,
    edgecolor='0.5'
)

sns.despine()

fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)

monthyearFmt = mdates.DateFormatter('%b %y')
ax.xaxis.set_major_formatter(monthyearFmt)

ax.plot(
    x_extrapolation,
    (lower_bound + 1)*yft.Close.iloc[-1],
    color='red'
)
ax.plot(
    x_extrapolation,
    (upper_bound + 1)*yft.Close.iloc[-1],
    color='green'
)
ax.fill_between(
    x_extrapolation,
    (lower_bound + 1)*yft.Close.iloc[-1],
    (upper_bound + 1)*yft.Close.iloc[-1],
    color='white', #'0.8',
    hatch='///',
    linewidth=0,
    edgecolor='0.5'
    )

lower_final_price = (lower_bound[-1] + 1) * yft.Close.iloc[-1]
upper_final_price = (upper_bound[-1] + 1) * yft.Close.iloc[-1]



ax.set_title(f"\${TICKER} (${yft.Close.iloc[-1]:.2f}) Expected Delta = {P_ITM} Move of {DTE} DTE based on prev {PERIOD}", size=22);
ax.set(
    xlim=(yft.index.min(), x_extrapolation[-1] + np.timedelta64(2, 'D')),
    ylim=(yft.Close.min()*0.98, 1.02*max(yft.Close.max(), ((upper_bound + 1)*yft.Close.iloc[-1]).max()))
)


ax.annotate(
    f"${lower_final_price:.2f}",
    xy=(x_extrapolation[-1] + np.timedelta64(2, 'D'), lower_final_price),
    size=20,
    va='center',
    color='red',
    backgroundcolor='white'
)

ax.annotate(
    f"${upper_final_price:.2f}",
    xy=(x_extrapolation[-1] + np.timedelta64(2, 'D'), upper_final_price),
    size=20,
    va='center',
    color='green',
    backgroundcolor='white'
)

plt.savefig("plot.png", transparent=False, facecolor='white', dpi=300)