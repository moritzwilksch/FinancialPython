# %%
from IPython.display import display, Markdown
import requests
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from requests.api import head, request
import seaborn as sns
import config
import random


api_key_fmp = random.choice(config.api_keys_fmp)

# %%

tickers = ['KR', 'WMT', 'TGT', 'COST']
# tickers = ['O', 'DLR', 'MPW', 'LTC']

LOAD_FROM_DISK = True

if not LOAD_FROM_DISK:
    for ticker in tickers:
        response = requests.get(f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={api_key_fmp}")
        joblib.dump(response, f"data/{ticker}_response.gz")
        print(f"[INFO] Got {ticker}.")


# response: requests.Response = joblib.load("data/response.gz")
responses = {ticker: joblib.load(f"data/{ticker}_response.gz") for ticker in tickers}

# %%
all_dfs = {ticker: pd.DataFrame(responses[ticker].json()) for ticker in tickers}
all_dfs = pd.concat(all_dfs)

all_dfs['date'] = all_dfs['date'].astype('datetime64')
all_dfs = (
    all_dfs
    .assign(year=all_dfs['date'].dt.year)
    .reset_index()
    .drop(['level_0', 'level_1'], axis=1)
)

years_as_sets = all_dfs.groupby('symbol')['year'].agg(set)
largest_common_year = max(years_as_sets.iloc[0].intersection(*years_as_sets.values))
# %%
mry = all_dfs.query("year == @largest_common_year").set_index('symbol')
mry

# %%
fields = [
    'grossProfitMargin',
    'netProfitMargin',
    'returnOnEquity',
    'returnOnCapitalEmployed',
    'dividendYield',
    'dividendPayoutRatio',
    'operatingCashFlowPerShare',
    'freeCashFlowPerShare',
    'debtEquityRatio',
    'cashPerShare',
    'priceEarningsRatio',
    'priceToFreeCashFlowsRatio',
    'priceFairValue',
]

rounding_rules = {
    'grossProfitMargin': 3,
    'netProfitMargin': 3,
    'returnOnEquity': 3,
    'returnOnCapitalEmployed': 3,
    'debtEquityRatio': 1,
    'operatingCashFlowPerShare': 2,
    'freeCashFlowPerShare': 2,
    'cashPerShare': 2,
    'dividendPayoutRatio': 3,
    'priceEarningsRatio': 1,
    'priceToFreeCashFlowsRatio': 1,
    'dividendYield': 3,
    'priceFairValue': 1,
}

pct_rows = [
    'grossProfitMargin',
    'netProfitMargin',
    'returnOnEquity',
    'returnOnCapitalEmployed',
    'dividendPayoutRatio',
    'dividendYield',
]

output = (
    mry[fields]
    .round(rounding_rules)
    .applymap(lambda s: f"{s:,}")
    .T
)

output.loc[pct_rows] = output.loc[pct_rows].applymap(lambda s: f"{float(s):.1%}")


# %%
try:
    response = requests.get(
        "https://sandbox.tradier.com/v1/markets/quotes",
        headers=config.tradier_headers,
        params={
            'symbols': ",".join(tickers)
        }
    )

    resp = response.json()

    prevclose = pd.DataFrame(resp.get('quotes').get('quote')).set_index('symbol')['prevclose'].to_frame().T
except:
    print("[ERROR] Could not fetch previous close.")
    prevclose = pd.Series(np.full(len(tickers), np.nan), index=tickers).to_frame().T

output = pd.concat((prevclose, output))

# %%
display(Markdown(output.to_markdown()))
with open("comparison.md", 'w') as f:
    f.write(output.to_markdown())

# %%
