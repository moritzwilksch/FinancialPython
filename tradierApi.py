# %%
import time
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import pandas as pd
import requests

headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer F8MghNUSQesfllneI0eod2TvqLjS'
    }

def get_iv(symbol: str, expiration: str) -> float:
    url = "https://sandbox.tradier.com/v1/markets/options/chains"
    payload = {
        'symbol': symbol,
        'expiration': expiration,
        'greeks': 'True'
    }

    response = requests.request("GET", url, headers=headers, params=payload)
    
    df = (pd.DataFrame(response.json()['options']['option']))
    df = (
        df
        .assign(iv=df.apply(lambda r: r['greeks'].get('mid_iv', 'NO'), axis=1))
    )

    current_underlying_price = 127.
    strikes = df['strike'].unique()
    closest_idx = np.abs((strikes - current_underlying_price)).argmin()
    used_strikes = strikes[closest_idx - 3: closest_idx+3]
    return df.loc[df.strike.isin(used_strikes)].iv.mean()

def get_expirations(symbol: str) -> List[str]:
    url = "https://sandbox.tradier.com/v1/markets/options/expirations"
    payload = {
        'symbol': symbol,
    }
    response = requests.request("GET", url, headers=headers, params=payload)

    return response.json()['expirations']['date']
#%%
expirations = get_expirations('aapl')

#%%
ivs = [get_iv('aapl', expiration=exp) for exp in expirations]

#%%
df = pd.DataFrame({
    'exp': expirations,
    'iv': ivs
})

#%%

fig, ax = plt.subplots()
sns.barplot(data=df, x='exp', y='iv', color='#00305e', ax=ax)
ax.set(ylim=(0,1))
plt.setp(ax.get_xticklabels(), rotation=45)
plt.show()

#%%