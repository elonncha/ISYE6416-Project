from data_preprocessing import *

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


meta = pd.read_excel('data/metadata.xlsx')
meta_latex = meta.iloc[:,0:2].to_latex(longtable = True)

with open('output/meta_latex.txt', 'w') as f:
    f.write(meta_latex)
    f.close()

data, target =  load_data(target = 'cancer', path = 'data/data_cleaned.csv', standardize = False)
descriptive_stats = data.describe().transpose().iloc[:,[0,1,2,3,5,7,]]

descriptive_stats = descriptive_stats.to_latex(float_format = '%.3f')

with open('output/data_descrip.txt', 'w') as f:
    f.write(descriptive_stats)
    f.close()

X,y = load_data(target = 'cancer', path = 'data/data_cleaned.csv', standardize = True)
X_train, X_test, y_train, y_test = split_dataset(X, y, train_fraction = 0.75)

# correlation
corr = X.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr, vmin=-1, vmax=1)
plt.savefig('output/corrplot.png')
plt.show()

mask = corr.where(corr > 0.7)


# scatterplot matrix
sns.set_theme(style="ticks")
sns.pairplot(X.iloc[:, [11, 12, 17, 18, 28]], diag_kind='kde')
plt.savefig('output/scatter.png')
plt.show()








#### ANALYSIS
pca = PCA(n_components = 13).fit(X.iloc[:,19:])

loading = pca.components_