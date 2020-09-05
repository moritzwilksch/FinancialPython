# %%
from keras_lr_finder import LRFinder
from tensorflow import keras
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
import yfinance

# %%
df = yfinance.Ticker('EURUSD=X').history(period='2y', interval='1h')

# %%


def start_pipeline(df):
    return df.copy()


def drop_useless(df):
    return df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)


def create_return(df):
    df['ret'] = df.Close / df.Open - 1
    return df


def create_simple_features(df):
    df['ma_8'] = df.ret.rolling(8).mean()
    df['sd_8'] = df.ret.rolling(8).std()
    df['ma_40'] = df.ret.rolling(40).mean()
    df['sd_40'] = df.ret.rolling(40).std()
    df['dayofweek'] = df.index.dayofweek

    for i in range(1, 8):
        df[f"ret_{i}"] = df.ret.shift(i)

    return df


def create_complex_features(df):
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['20dSTD'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['ma_20'] + (df['20dSTD'] * 2)
    df['Lower'] = df['ma_20'] - (df['20dSTD'] * 2)
    df['dist_to_Upper'] = df['Upper'] - df['Close']
    df['dist_to_Lower'] = df['Lower'] - df['Close']
    return df


def dropna(df):
    return df.dropna()


def create_target(df):
    df['target'] = df.ret.shift(-1)
    return df


def tt_split(df, trainpct=0.8):
    trainidx = int(len(df)*trainpct)
    xtrain = df.drop('target', axis=1).iloc[:trainidx]
    ytrain = df.target[:trainidx]
    xtest = df.drop('target', axis=1).iloc[trainidx:]
    ytest = df.target[trainidx:]

    return xtrain, xtest, ytrain, ytest


def binarize_target(series, thresh=0):
    if thresh == 0:
        return (series > 0).astype('int8')
    else:
        return series.map(lambda v: 1 if v > thresh else -1 if v < -thresh else 0).astype('int8')


# %%
preped_df = (df
             .pipe(start_pipeline)
             .pipe(drop_useless)
             .pipe(create_return)
             .pipe(create_simple_features)
             .pipe(create_complex_features)
             .pipe(dropna)
             .pipe(create_target)
             )

xtrain, xtest, ytrain, ytest = tt_split(preped_df, trainpct=0.85)
xtrain, xval, ytrain, yval = tt_split(pd.concat([xtrain, ytrain], axis=1))

ss = StandardScaler()


def scale(df, fit=False, cols_to_scale=None):
    if cols_to_scale is None:
        cols_to_scale = df.columns

    if fit:
        ss.fit(df[cols_to_scale])
    scaled = ss.transform(df[cols_to_scale])
    df[cols_to_scale] = scaled
    return df


# Scale features for Neural Nets!
xtrain = xtrain.pipe(
    scale, fit=True, cols_to_scale=xtrain.columns.drop('dayofweek').tolist())
xval = xval.pipe(scale, fit=False,
                 cols_to_scale=xtrain.columns.drop('dayofweek').tolist())
xtest = xtest.pipe(
    scale, fit=False, cols_to_scale=xtrain.columns.drop('dayofweek').tolist())

# %%
for x in [xtrain, xval, xtest]:
    print(x.shape)

threshold = 0  # ytrain.std()/2
ytrain = binarize_target(ytrain, thresh=threshold)
yval = binarize_target(yval, thresh=threshold)
ytest = binarize_target(ytest, thresh=threshold)
# %%

rf = GridSearchCV(
    RandomForestClassifier(n_jobs=-1),
    param_grid={
        'min_samples_leaf': [3, 5, 7, 10],
        'max_features': ['auto', 'sqrt', 0.5],
        'max_depth': [5, 10, 15]
    }
)

rf = RandomForestClassifier(max_depth=5, min_samples_leaf=3)
rf.fit(xtrain, ytrain)
# %%
print("\n" + classification_report(ytest, rf.predict(xtest)))
print(confusion_matrix(ytest, rf.predict(xtest)))

pd.Series(rf.feature_importances_,
          index=xtrain.columns).sort_values().plot(kind='barh')
plt.title("RandomForest Feature Importances")
# %%
cbc = CatBoostClassifier(eval_metric='Accuracy', verbose=False)
cbc.fit(xtrain, ytrain, eval_set=[(xval, yval)])

print("\n" + classification_report(ytest, cbc.predict(xtest)))
print(confusion_matrix(ytest, cbc.predict(xtest)))

# %%
pd.Series(cbc.feature_importances_,
          index=xtrain.columns).sort_values().plot(kind='barh')
plt.title("CatBoost Feature Importances")

# %%
nn = keras.Sequential([
    keras.layers.Dense(units=xtrain.shape[1]*2, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(units=int(xtrain.shape[1]/2), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(units=1, activation='sigmoid')
]
)
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
lr_finder = LRFinder(nn)
lr_finder.find(xtrain.values, ytrain.values, start_lr=0.0001,
               end_lr=1, batch_size=512, epochs=5)
lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
plt.show()
lr_finder.plot_loss_change(sma=20, n_skip_beginning=20,
                           n_skip_end=5, y_lim=(-0.01, 0.01))
plt.show()
# %%





# %%
n_emb_fts = 1
inp_normal = keras.layers.Input(
    shape=(xtrain.shape[1] - n_emb_fts, ), name='inp_normal')
inp_dow_embedding = keras.layers.Input(
    shape=(n_emb_fts, ), name='inp_dow_embedding')
dow_embedding = keras.layers.Embedding(
    input_dim=5, output_dim=3, input_length=1)(inp_dow_embedding)
dow_embedding = keras.layers.Flatten()(dow_embedding)
concat = keras.layers.Concatenate()([inp_normal, dow_embedding])

x = keras.layers.Dense(units=40, activation='relu', kernel_regularizer='l2')(concat)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(units=40, activation='relu', kernel_regularizer='l2')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(units=20, activation='relu', kernel_regularizer='l2')(x)
x = keras.layers.BatchNormalization()(x)
out = keras.layers.Dense(units=1, activation='sigmoid')(x)


nn = keras.Model(inputs=[inp_normal, inp_dow_embedding], outputs=out)
nn.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=10**-1.9), metrics=['accuracy'])

h = nn.fit(
    x={'inp_normal': xtrain.drop('dayofweek', axis=1).values,
       'inp_dow_embedding': xtrain.dayofweek.values.reshape(-1, 1)},
    y=ytrain.values,
    validation_data=(
        [xval.drop('dayofweek', axis=1).values,
         xval.dayofweek.values.reshape(-1, 1)],
        yval.values
    ),
    epochs=20,
    batch_size=64
    )

print("\n" + classification_report(ytest, nn([xtest.drop('dayofweek', axis=1).values, xtest.dayofweek.values])>0.5))
print(confusion_matrix(ytest, nn([xtest.drop('dayofweek', axis=1).values, xtest.dayofweek.values])>0.5))

pd.DataFrame({'train': h.history['loss'], 'val': h.history['val_loss']}).plot()

#%%
from lr_finder import LRFinder
lr_finder = LRFinder(nn)
lr_finder.find([xtrain.drop('dayofweek', axis=1).values, xtrain.dayofweek.values], ytrain.values, start_lr=0.0001,
               end_lr=1, batch_size=1024, epochs=5)
lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
plt.show()
lr_finder.plot_loss_change(sma=20, n_skip_beginning=20,
                           n_skip_end=5, y_lim=(-0.01, 0.01))
plt.show()
# %%
def prep_for_sequence_model(df, timesteps=5):
    x = []
    for lower_i in range(0, len(df)-timesteps):
        x.append(df.iloc[lower_i:lower_i+timesteps].values)

    return np.array(x)


timesteps = 5

xtrain_seq = prep_for_sequence_model(xtrain, timesteps)
ytrain_seq = ytrain[timesteps:]

xval_seq = prep_for_sequence_model(xval, timesteps)
yval_seq = yval[timesteps:]

xtest_seq = prep_for_sequence_model(xtest, timesteps)
ytest_seq = ytest[timesteps:]

# %%
seq_nn = keras.Sequential([
    keras.layers.GRU(input_shape=(timesteps, 22),
                     units=25, return_sequences=True),
    keras.layers.GRU(units=10),
    keras.layers.Dense(units=1, activation='sigmoid')

])

seq_nn.compile(loss='binary_crossentropy',
               optimizer='adam', metrics=['accuracy'])
seq_nn.fit(xtrain_seq, ytrain_seq, validation_data=(
    xval_seq, yval_seq), epochs=10, batch_size=32)




#%%