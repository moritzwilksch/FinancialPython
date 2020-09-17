# %%
from torch_lr_finder import LRFinder  # this is the PyPi package
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import yfinance

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
    return (((series+1).rolling(n).apply(np.prod, raw=True).shift(-n-1)-1) > 0).astype('int8')


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

# %%
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

for x in [xtrain, xval, xtest]:
    print(x.shape)

ytrain = prep_target_next_n(ytrain, n=3)
yval = prep_target_next_n(yval, n=3)
ytest = prep_target_next_n(ytest, n=3)

eye = torch.eye(2)

xtrain = torch.Tensor(xtrain.values)
ytrain = torch.Tensor(ytrain.values)

xval = torch.Tensor(xval.values)
yval = torch.Tensor(yval.values)

xtest = torch.Tensor(xtest.values)
ytest = torch.Tensor(ytest.values)

from torch.utils.data import WeightedRandomSampler
weights = torch.zeros((xtrain.size(0)))
weights[ytrain.bool()] = 1/torch.sum(ytrain.float())
weights[~ytrain.bool()] = 1/(xtrain.size(0) - torch.sum(ytrain.float()))
rand_sampler = WeightedRandomSampler(weights=weights, num_samples = xtrain.size(0))

BATCHSIZE = 32
train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=BATCHSIZE, sampler=rand_sampler)
val_loader = DataLoader(TensorDataset(xval, yval), shuffle=False)
test_loader = DataLoader(TensorDataset(xtest, ytest), shuffle=False)

idx_to_be_embedded = [preped_df.columns.get_loc('dayofweek')]
emb_mask = torch.zeros((xtrain.size(1), )).bool()
emb_mask[idx_to_be_embedded] = True
# %%
from sklearn.metrics import precision_score
class MyNet(nn.Module):
    def __init__(self, h1=100, h2=40, h3=10, dow_embds_size=3, dropout=0.3):
        super(MyNet, self).__init__()
        self.dow_embedding = nn.Embedding(7, dow_embds_size)
        self.input = nn.Linear(in_features=xtrain.size(1) + dow_embds_size - len(idx_to_be_embedded), out_features=h1)
        self.bn1 = nn.BatchNorm1d(num_features=h1)
        self.hidden1 = nn.Linear(in_features=h1, out_features=h2)
        self.bn2 = nn.BatchNorm1d(num_features=h2)
        self.hidden2 = nn.Linear(in_features=h2, out_features=h3)
        self.bn3 = nn.BatchNorm1d(num_features=h3)
        self.out = nn.Linear(in_features=h3, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dow_embds=None):
        # Extract cols to be embedded from x and delete them from x
        dow_embds = x[:, emb_mask].long()
        x = x[:, ~emb_mask]

        # Input preparation
        dow_embds = self.dow_embedding(dow_embds)

        # problem without view: torch.Size([8, 1, 3]) & torch.Size([8, 42])
        concated = torch.cat((x, dow_embds.view(dow_embds.size(0), dow_embds.size(2))), dim=1)

        # Hidden layers
        x = self.input(concated)
        x = self.dropout(self.bn1(self.relu(x)))
        x = self.hidden1(x)
        x = self.dropout(self.bn2(self.relu(x)))
        x = self.hidden2(x)
        x = self.dropout(self.bn3(self.relu(x)))
        x = self.out(x)
        out = self.sigmoid(x)
        return out

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

#%%
net = MyNet(100, 50, 20, 3, 0.3)
criterion = nn.BCELoss()
optim = torch.optim.Adam(net.parameters(), lr=10**-2)
# Explicitly init weights!
net.apply(init_weights)

# %%
lrf = LRFinder(net, optim, criterion)
lrf.range_test(train_loader, start_lr=0.0001, end_lr=1)
lrf.plot()
lrf.reset()

#%%
# seemingly best: Adam + cyclical LR
N_EPOCHS = 25
scheduler = torch.optim.lr_scheduler.CyclicLR(optim, 10**-2, 10**-1, mode='triangular2', step_size_up=(xtrain.size(0)/BATCHSIZE)*2, cycle_momentum=False)

history = {'train_loss': [], 'val_loss': []}
for epoch in range(N_EPOCHS):
    _losses = []
    for x, y in train_loader:
        yhat = net(x)
        loss = criterion(yhat, y)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        with torch.no_grad():
            _losses.append(loss.item())

    with torch.no_grad():
        val_loss = criterion(net(xval), yval)
        precision = precision_score(yval, net(xval) > 0.5)
        print(f"EPOCH {epoch:4}: train_loss = {np.mean(_losses):.3f} | val_loss = {val_loss.item():.3f} | val_prec = {precision:.2f}")
        history['train_loss'].append(np.mean(_losses).item())
        history['val_loss'].append(val_loss.item())

pd.DataFrame(history).plot()

# %%
preds = net(xval).detach().numpy().flatten()
print("\n" + classification_report(yval, preds > 0.5))
print(f"{confusion_matrix(yval, preds > 0.5)}")

from sklearn.metrics import precision_recall_curve
prc = precision_recall_curve(yval, preds)
print(f"Maximum Precision Threshold = {np.max(prc[0][:-1]):.4f}")
max_prec_thresh = prc[2][np.argmax(prc[0][:-1]).flat]
print("\n" + classification_report(yval, preds > max_prec_thresh))
print(confusion_matrix(yval, preds > max_prec_thresh))