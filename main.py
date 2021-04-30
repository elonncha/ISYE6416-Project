from data_preprocessing import *

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline


from sklearn.decomposition import PCA





X,y = load_data(target = 'cancer', path = 'data/data_cleaned.csv', standardize = True)
X_train, X_test, y_train, y_test = split_dataset(X, y, train_fraction = 0.75)





# correlation
corr = X.corr()


plt.figure(figsize=(16, 12))
sns.heatmap(loading, vmin=-1, vmax=1)
plt.show()


pca = PCA(n_components = 13).fit(X.iloc[:,19:])

loading = pca.components_