# %%
from sklearn.metrics import precision_recall_curve
from CLRCallback import CyclicLR
from lr_finder import LRFinder
from tensorflow import keras
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import yfinance
import tensorflow as tf
tf.random.set_seed(1234)

# %%
df = yfinance.Ticker('EURUSD=X').history(period='5y')

# %%
def start_pipeline(df):
    return df.copy()


def drop_useless(df):
    return df.drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)


def create_return(df):
    df['ret'] = df.Close / df.Open - 1
    return df


def create_simple_features(df):
    """df['ma_8'] = df.ret.rolling(8).mean()
    df['pct_ma_8'] = df.ret.rolling(8).mean()
    df['sd_8'] = df.ret.rolling(8).std()
    df['ma_40'] = df.ret.rolling(40).mean()
    df['sd_40'] = df.ret.rolling(40).std()"""
    df['dayofweek'] = df.index.dayofweek.astype('int8')
    df['hourofday'] = df.groupby(df.index)['Open'].transform(lambda s: list(range(len(s)))).astype('int8')

    for i in range(2, 11):
        df[f"ma_{i}"] = df.ret.rolling(i).mean()
        df[f"sd_{i}"] = df.ret.rolling(i).std()

    for i in [2, 5, 7]:
        df[f"ma{i}_greater_ma10"] = (df[f"ma_{i}"] > df['ma_10']).astype('uint8')

    for i in range(1, 10):
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
    xval = df.drop('target', axis=1).iloc[trainidx:]
    yval = df.target[trainidx:]

    return xtrain, xval, ytrain, yval


def binarize_target(series, thresh=0):
    if thresh == 0:
        return (series > 0).astype('int8')
    else:
        return series.map(lambda v: 1 if v > thresh else -1 if v < -thresh else 0).astype('int8')


def prep_target_next_n(series: pd.Series, n: int):
    return (series.rolling(n).sum().shift(-n-1) > 0).astype('int8')


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
cat_cols = ['dayofweek', 'hourofday']

xtrain = xtrain.pipe(scale, fit=True, cols_to_scale=xtrain.columns.drop(cat_cols).tolist())
xval = xval.pipe(scale, fit=False, cols_to_scale=xtrain.columns.drop(cat_cols).tolist())
xtest = xtest.pipe(scale, fit=False, cols_to_scale=xtrain.columns.drop(cat_cols).tolist())

# %%
for x in [xtrain, xval, xtest]:
    print(x.shape)

threshold = 0  # ytrain.std()/2
"""ytrain = binarize_target(ytrain, thresh=threshold)
yval = binarize_target(yval, thresh=threshold)
ytest = binarize_target(ytest, thresh=threshold)"""

ytrain = prep_target_next_n(ytrain, n=3)
yval = prep_target_next_n(yval, n=3)
ytest = prep_target_next_n(ytest, n=3)

# %%

rf = GridSearchCV(
    RandomForestClassifier(n_jobs=-1),
    param_grid={
        'min_samples_leaf': [3, 5, 7, 10, 25],
        'max_features': ['auto', 'sqrt', 0.5, 0.7],
        'max_depth': [5, 10, 15, 20, 25]
    }
)

rf = RandomForestClassifier(max_depth=5, min_samples_leaf=7)
rf.fit(xtrain, ytrain)

print("\n" + classification_report(yval, rf.predict(xval)))
print(confusion_matrix(yval, rf.predict(xval)))

pd.Series(rf.feature_importances_, index=xtrain.columns).sort_values().plot(kind='barh')
plt.title("RandomForest Feature Importances")
# %%
cbc = CatBoostClassifier(eval_metric='Precision', verbose=False, l2_leaf_reg=1)
cbc.fit(xtrain, ytrain, eval_set=[(xval, yval)], cat_features=cat_cols)

print("\n" + classification_report(yval, cbc.predict(xval)))
print(confusion_matrix(yval, cbc.predict(xval)))

print(f"Maximum Precision Threshold")
prc = precision_recall_curve(yval, cbc.predict_proba(xval)[:, -1])
max_prec_thresh = prc[2][np.argmax(prc[0][:-1]).flat]
print("\n" + classification_report(yval, cbc.predict_proba(xval)[:, -1] > max_prec_thresh))
print(confusion_matrix(yval, cbc.predict_proba(xval)[:, -1] > max_prec_thresh))

# %%
pd.Series(cbc.feature_importances_, index=xtrain.columns).sort_values().plot(kind='barh')
plt.title("CatBoost Feature Importances")


# %%
###############################################################################
###############################################################################
embedding_features = ['dayofweek', 'hourofday']


def get_model(batchsize=8, dropout=0.3):
    # Inputs
    inp_normal = keras.layers.Input(shape=(xtrain.shape[1] - len(embedding_features), ), name='inp_normal')
    inp_dow_embedding = keras.layers.Input(shape=(1, ), name='inp_dow_embedding')
    inp_hod_embedding = keras.layers.Input(shape=(1, ), name='inp_hod_embedding')

    # Embeddings
    dow_embedding = keras.layers.Embedding(input_dim=7, output_dim=3, input_length=1)(inp_dow_embedding)
    dow_embedding = keras.layers.Flatten()(dow_embedding)

    hod_embedding = keras.layers.Embedding(input_dim=24, output_dim=10, input_length=1)(inp_hod_embedding)
    hod_embedding = keras.layers.Flatten()(hod_embedding)

    # Hidden layers
    concat = keras.layers.Concatenate()([inp_normal, dow_embedding, hod_embedding])
    x = keras.layers.Dense(units=100, activation='relu')(concat)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(units=40, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(units=10, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(units=1, activation='sigmoid')(x)

    nn = keras.Model(inputs=[inp_normal, inp_dow_embedding, inp_hod_embedding], outputs=out)
    nn.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.001),
        metrics=['accuracy', keras.metrics.Precision()]
    )

    lr_finder = LRFinder(0.0001, 0.1)
    nn.fit(
        x={
            'inp_normal': xtrain.drop(embedding_features, axis=1).values,
            'inp_dow_embedding': xtrain.dayofweek.values.reshape(-1, 1),
            'inp_hod_embedding': xtrain.hourofday.values.reshape(-1, 1)
        },
        y=ytrain.values,
        validation_data=(
            [
                xval.drop(embedding_features, axis=1).values,
                xval.dayofweek.values.reshape(-1, 1),
                xval.hourofday.values.reshape(-1, 1)
            ],
            yval.values
        ),
        epochs=2,
        batch_size=batchsize,
        callbacks=[lr_finder]
    )
    return nn


# %%
# Training metrics worse than validation metrics because of dropout
BATCHSIZE = 8
nn = get_model(batchsize=BATCHSIZE, dropout=0.3)  # Experiment with dropout to make LR-Finder well-behaved

# %%
cycle_lr = CyclicLR((10**-4), 10**-3.9, mode='exp_range')  # exp_range works much better than triangular
h = nn.fit(
    x={
        'inp_normal': xtrain.drop(embedding_features, axis=1).values,
        'inp_dow_embedding': xtrain.dayofweek.values.reshape(-1, 1),
        'inp_hod_embedding': xtrain.hourofday.values.reshape(-1, 1),
    },
    y=ytrain.values,
    validation_data=(
        [
            xval.drop(embedding_features, axis=1).values,
            xval.dayofweek.values.reshape(-1, 1),
            xval.hourofday.values.reshape(-1, 1)
        ],
        yval.values
    ),
    epochs=15,
    batch_size=BATCHSIZE,
    callbacks=[cycle_lr]
)

# %%

print("\n" + classification_report(
    yval.values,
    nn.predict([xval.drop(embedding_features, axis=1).values, xval.dayofweek.values, xval.hourofday.values]) > 0.5
))
print(confusion_matrix(yval, nn.predict([xval.drop(embedding_features, axis=1).values, xval.dayofweek.values, xval.hourofday.values]) > 0.5))

pd.DataFrame({'train': h.history['loss'], 'val': h.history['val_loss']}).plot()

# %%
print(f"Maximum Precision Threshold = {np.max(prc[0][:-1]):.4f}")
prc = precision_recall_curve(yval, nn.predict([xval.drop(embedding_features, axis=1).values, xval.dayofweek.values, xval.hourofday.values]))
max_prec_thresh = prc[2][np.argmax(prc[0][:-1]).flat]
print("\n" + classification_report(yval.values, nn.predict([xval.drop(embedding_features, axis=1).values, xval.dayofweek.values, xval.hourofday.values]) > max_prec_thresh))
print(confusion_matrix(yval, nn.predict([xval.drop(embedding_features, axis=1).values, xval.dayofweek.values, xval.hourofday.values]) > max_prec_thresh))
