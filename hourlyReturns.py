# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance

# %%
df = yfinance.Ticker('EURUSD=X').history(period='1y', interval='1h')
df = df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)
df['ret'] = df.Close / df.Open - 1
df['next_ret'] = df.shift(-1)['ret']
df['next_ret_bin'] = df['next_ret'] > 0
df = df.dropna()
df['ma200'] = df.rolling(200).mean()['Close']
df['ma50'] = df.rolling(50).mean()['Close']
df['ma_above'] = df.ma50 > df.ma200
df['closed_above_ma'] = df['Close'] > df.ma50
df['MAC'] = df['ma200'] - df['ma50']
df = df.dropna()
# %%
plt.figure(figsize=(20, 5))
plt.plot(df.Close, color='b')
plt.plot(df['ma200'], color='red')
plt.plot(df['ma50'], color='g')
plt.show()

# %%
plt.figure(figsize=(20, 5))
plt.plot(df.ret, color='b')
plt.show()

# %%
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(df.drop(['next_ret', 'next_ret_bin', 'Open', 'High', 'Low', 'Close'],
                                                        axis=1),
                                                df.next_ret_bin,
                                                shuffle=False)
xtrain = xtrain.values
xtest = xtest.values
ytrain = ytrain.values
ytest = ytest.values

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import keras
model = keras.Sequential([
    keras.layers.Dense(units=100, activation='relu'),
    keras.layers.Dense(units=100, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile('adam', 'binary_crossentropy')
model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=10)

preds = model.predict_classes(xtest)
print(confusion_matrix(ytest, preds))

#%%

