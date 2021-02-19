# %%
import pandas as pd
import requests
url = "https://api.pushshift.io/reddit/search/submission?subreddit=wallstreetbets&size=1000&selftext=GME"

# %%
response = requests.get(url)

# %%
rawdf: pd.DataFrame = pd.DataFrame(response.json()['data'])

#%%
rawdf.info()

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
mat = tfidf.fit_transform(rawdf.selftext)

#%%
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=3)
lda.fit(mat)

#%%
lda.transform(mat).argmax(axis=1)