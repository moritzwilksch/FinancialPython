# %%
from joblib import Parallel, delayed
import numba
import numpy as np
from typing import List
import pandas as pd
from pandas.io.parquet import to_parquet
import requests
import yfinance as yf
# %%
tickers = requests.get("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents_symbols.txt").text.split()

# %%
SCRAPE_FROM_WEB = False

if SCRAPE_FROM_WEB:
    dfs = {}
    for ticker in tickers:
        print(ticker)
        dfs.update({ticker: yf.Ticker(ticker).history(start="2008-01-01", end="2021-01-01")['Close']})
    data = pd.concat(dfs, axis=1)
    # data.to_parquet("sp500_history.parquet")
else:
    data = pd.read_parquet("sp500_history.parquet")

# %%
returns: pd.DataFrame = data.pct_change().fillna(0)
# %%
idx_mapping = {t: idx for idx, t in enumerate(data.columns.values)}

# @numba.njit


def get_portfolio(size: int) -> List[str]:
    return np.random.choice(tickers, size=size, replace=False)

# @numba.jit


def get_portfolio_std(portfolio: np.array):
    return ((returns[portfolio]+1).cumprod(axis=0).sum(axis=1)/len(portfolio)).std()


# %%
SIZE = 25
N = 1000
# results = {s: [] for s in range(1, SIZE)}
# for s in range(1, SIZE):
#     for n in range(N):
#         results[s].append(get_portfolio_std(get_portfolio(size=s)))

#%%
def simulate(s, N):
    l = []
    for n in range(N):
        l.append(get_portfolio_std(get_portfolio(size=s)))
    return l
#%%
from functools import partial
results = Parallel(n_jobs=-1, prefer='processes')(delayed(partial(simulate, N=7500))(s) for s in range(1, 15))
#%%
import seaborn as sns
f = pd.DataFrame(results).T.mean().to_frame()
sns.lineplot(f.index, f.values.flat, color='blue', lw=1.5)
sns.despine()