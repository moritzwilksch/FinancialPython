# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.stats as stats

# %%
portfolio = ["ALV", "T", "AAPL", "EVD.DE", "DLR", "O", "KRN.DE", "RYAAY", "VEUR.AS", "IQQ6.DE", "SPY", "BRK-B", "EQR", "JNJ", "UL", "SPY"]
dfs = []

yf_tickers_portfolio = [yf.Ticker(symbol) for symbol in portfolio]

for ticker in yf_tickers_portfolio:
    dfs.append(pd.DataFrame(ticker.history("5y")["Close"]).rename({"Close": (ticker.ticker)}, axis=1))
    print(f"Fetched {ticker}!")

stocks = pd.concat(dfs, axis=1)

# Create Returns DF
returns = stocks.pct_change()

# %%
fig, axes = plt.subplots(1, 1, figsize=(15, 13))
sns.heatmap(returns.corr(), annot=True, ax=axes, cmap="coolwarm")
plt.title("Portfolio Correlation", size=18);

#%%
potentials = ["GOOG", 'FB', 'AAPL']
yf_tickers_potentials = yf.Tickers(' '.join(potentials))

dfs_potentials = []
for ticker in yf_tickers_potentials.tickers:
    dfs_potentials.append(pd.DataFrame(ticker.history("5y")["Close"]).rename({"Close": (ticker.ticker)}, axis=1))
    print(f"Fetched {ticker}!")

prices_potentials = pd.concat(dfs_potentials, axis=1)
returns_potentials = prices_potentials.pct_change()

#%%

fig, axes = plt.subplots(1, 1, figsize=(15, 13))
sns.heatmap(pd.concat([returns, returns_potentials], axis=1).corr(), annot=True, ax=axes, cmap="coolwarm")
plt.title("Portfolio Correlation + POTENTIALS", size=18);
plt.axvline(x=len(portfolio), color='k', linewidth=2.5);
plt.axhline(y=len(portfolio), color='k', linewidth=2.5);
