# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn.utils import despine
import yfinance as yf
import scipy.stats as stats

# %%
portfolio = ["T", "AAPL", "DLR", "O", "KRN.DE", "VEUR.AS", "BRK-B", "EQR", "JNJ", "UL", "SPY", "MMM", "BTI", "GOOD", "IBM", "LTC", "ORCC", "PLTR", "PG", "VZ", 'ISPA.DE', 'IQQ6.DE', 'QYLD', 'MPW']
dfs = []

yf_tickers_portfolio = [yf.Ticker(symbol) for symbol in portfolio]

for ticker in yf_tickers_portfolio:
    dfs.append(pd.DataFrame(ticker.history("5y")["Close"]).rename({"Close": (ticker.ticker)}, axis=1))
    print(f"Fetched {ticker}!")

min_df_len = min([len(df) for df in dfs])
stocks = pd.concat([df.iloc[-min_df_len:, :].dropna() for df in dfs], axis=1)

#%%
# Create Returns DF
returns = stocks.pct_change()

# %%
fig, axes = plt.subplots(1, 1, figsize=(15, 13))
sns.heatmap(returns.corr(), annot=True, ax=axes, cmap="coolwarm")
plt.title("Portfolio Correlation", size=18);


#%%
# sns.clustermap(returns.fillna(0).T)
#%%
potentials = ["SIX2.DE", 'BLK', 'LTC', 'ISPA.DE', 'IQQ6.DE', 'QYLD', 'MPW', 'ARCC', 'RYLD']
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

#%%
# Historical Performance
fig, ax = plt.subplots(figsize=(10,5))
for col in stocks:
    color = 'blue' if col == "AAPL" else '0.2'
    ax.plot(stocks[col]/stocks.loc["2015-11-23", col], color=color, linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("gaph.png")
