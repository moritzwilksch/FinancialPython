# %%
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

# %%
portfolio = ["T", "AAPL", "DLR", "O", "KRN.DE", "VEUR.AS", "BRK-B", "EQR", "JNJ", "UL", "SPY", "MMM",
             "BTI", "GOOD", "IBM", "LTC", "ORCC", "PLTR", "PG", "VZ", 'ISPA.DE', 'IQQ6.DE', 'QYLD', 'MPW']
dfs = []

yf_tickers_portfolio = [yf.Ticker(symbol) for symbol in portfolio]

for ticker in yf_tickers_portfolio:
    dfs.append(pd.DataFrame(ticker.history("5y")["Close"]).rename({"Close": (ticker.ticker)}, axis=1))
    print(f"Fetched {ticker}!")

min_df_len = min([len(df) for df in dfs])
stocks = pd.concat([df.iloc[-min_df_len:, :].dropna() for df in dfs], axis=1)

#%%
# Create Returns DF
returns = stocks.pct_change().dropna().T.drop('PLTR')

# %%
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

errors = []
for n in range(2, 20):
    agg = KMeans(n_clusters=n)
    preds = agg.fit_predict(returns)
    err = agg.inertia_#returns.groupby(preds).mean().std().mean()
    errors.append((n, err))
#%%
pd.DataFrame(errors).set_index(0).plot()

#%%
agg = KMeans(n_clusters=8)
preds = agg.fit_predict(returns)
res = returns.groupby(preds).groups
res

#%%
from sklearn.decomposition import PCA

evs = []
for nc in range(1, 24, 2):
    p = PCA(n_components=nc)
    p.fit(returns)
    evs.append((nc, p.explained_variance_.sum()))

#%%%%
pd.DataFrame(evs).set_index(0).plot()


#%%
agg = DBSCAN()
preds = agg.fit_predict(returns)
res = returns.groupby(preds).groups
res

#%%
agg = KMeans()
preds = agg.fit_predict(returns)
res = returns.groupby(preds).groups
res
