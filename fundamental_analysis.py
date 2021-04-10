#%%
from math import log, floor
from matplotlib import markers
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titleweight'] = 'bold'

def millions(x, pos=None):
    """The two args are the value and tick position."""
    return '${:1.0f}M'.format(x*1e-6)


BLUE = '#004BA8'
#%%
ticker = yf.Ticker("AAPL")

ocf = ticker.cashflow.T['Total Cash From Operating Activities'].sort_index()
capex = ticker.cashflow.T['Capital Expenditures'].sort_index()
fcf = ocf + capex  # capex are already negative
fcf_series = fcf.reindex(pd.date_range(start=ocf.index.min(), end=history.index.max())).ffill()

info = ticker.get_info() # TODO

history = ticker.history(start=ocf.index.min())['Close']


#%%
import seaborn as sns
fig, ax = plt.subplots(figsize=(10,7))

years = fcf.index.year
bars = ax.bar(years, fcf, color=BLUE, ec='k')

ax.set_title(f"FCF from {years.min()} - {years.max()}", size=24)
ax.set(
    xticks=years,
    ylim=(0, fcf.max()*1.1),
    ylabel="FCF"
);

ax.bar_label(bars, labels=[millions(x) for x in fcf], padding=10, size=16, weight='bold')
ax.yaxis.set_major_formatter(millions)
sns.despine(ax=ax)

plt.show()


#%%
pcf = history/(fcf_series/info['sharesOutstanding'])

fig, ax = plt.subplots(figsize=(15,5))
sns.lineplot(x=pcf.index, y=pcf, ax=ax, color=BLUE)
ax.set_title(f"Price/FCF from {years.min()} - {years.max()}", size=24)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax.set_ylabel("Price/FCF")
ax.text(
    x=pcf.index.max() + pd.to_timedelta(1, "w"),
    y=pcf[-1],
    s=f"{pcf[-1]:.1f}",
    weight='bold',
    bbox={'boxstyle': 'square', 'fill': None}
    )
sns.despine()


#%%
from rich.table import Table
from rich.console import Console
c = Console()
tab = Table()
tab.add_column('Metric')
tab.add_column('Value')
tab.add_row("M1", "42")
c.print(tab)