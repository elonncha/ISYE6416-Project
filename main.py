from data_preprocessing import *

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor


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
X_train, X_test, y_train, y_test = split_dataset(X, y, train_fraction=0.75)

# correlation
corr = X.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr, vmin=-1, vmax=1)
#plt.savefig('output/corrplot.png')
plt.show()

mask = corr.where(corr > 0.7)


# scatterplot matrix
sns.set_theme(style="ticks")
sns.pairplot(X.iloc[:, [11, 12, 17, 18, 28]], diag_kind='kde')
#plt.savefig('output/scatter.png')
plt.show()








#### ANALYSIS
# 1. random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import graphviz
OOB, Test = [], []
for d in ['cancer', 'chd', 'mental']:
    data, target = load_data(target=d, path='data/data_cleaned.csv', standardize=True)
    X_train, X_test, y_train, y_test = split_dataset(X, y, train_fraction=0.75)
    for n_tree in [50,100]:
        RFG = RandomForestRegressor(n_estimators=n_tree, min_samples_split = 20, oob_score = True)
        RFG.fit(X_train, y_train)
        OOB.append(RFG.oob_score_)
        Test.append(RFG.score(X_test, y_test))


RFG = RandomForestRegressor(n_estimators=20, min_samples_split = 20, oob_score = True)
RFG.fit(X_train, y_train)

forest_importances = pd.Series(RFG.feature_importances_, index=X_train.columns)
std = np.std([RFG.feature_importances_ for tree in RFG.estimators_], axis=0)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig('output/rf1.png')
plt.show()



RFG2 = RandomForestRegressor(n_estimators=20, min_samples_split = 20, oob_score = True)
RFG2.fit(X_train.drop(columns = ['BPMED']), y_train)

forest_importances = pd.Series(RFG2.feature_importances_, index= X_train.drop(columns = ['BPMED']).columns)
std = np.std([RFG.feature_importances_ for tree in RFG2.estimators_], axis=0)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig('output/rf2.png')
plt.show()


RFG3 = RandomForestRegressor(n_estimators=50, oob_score = True, min_samples_leaf = 200, max_depth=20)
RFG3.fit(X_train.loc[:, ['MINORPCT', 'OVER64PCT', 'OZONE', 'PM25', 'BINGE', 'CHECKUP']], y_train)

graph_data = tree.export_graphviz(RFG3.estimators_[0],
                                  feature_names= ['MINORPCT', 'OVER64PCT', 'OZONE', 'PM25', 'BINGE', 'CHECKUP'],
                                  class_names = ['MINORPCT', 'OVER64PCT', 'OZONE', 'PM25', 'BINGE', 'CHECKUP'],
                                  filled = True, rounded=True, special_characters=True)
graph = graphviz.Source(graph_data)
graph.view()

RFG3.oob_score_





# 2. ANN


loss = []
for i in tqdm(range(20)):
    ANN1 = MLPRegressor(hidden_layer_sizes = (128,), alpha = 0.001, batch_size = 128, learning_rate = 'adaptive')
    ANN1.fit(X_train,y_train)
    loss.append(ANN1.loss_curve_)




loss2 = []
for i in tqdm(range(20)):
    ANN2 = MLPRegressor(hidden_layer_sizes = (128,64), alpha = 0.001, batch_size = 128, learning_rate = 'adaptive')
    ANN2.fit(X_train,y_train)
    loss2.append(ANN2.loss_curve_)


plt.figure()
for i in range(20):
    plt.plot(range(len(loss[i])), loss[i], alpha = 0.25, color = 'blue', label = '1-layer')

plt.xlabel('EPOCH')
plt.ylabel('MSE loss')
plt.savefig('output/loss1.png')
plt.show()


plt.figure()
for i in range(20):
    plt.plot(range(len(loss2[i])), loss2[i], alpha = 0.25, color = 'red', label = '2-layer')
plt.xlabel('EPOCH')
plt.ylabel('MSE loss')
plt.savefig('output/loss2.png')
plt.show()


loss, score = [], []
n = 16
for a in tqdm(np.logspace(-5,2, n, endpoint=True)):
    ANN1 = MLPRegressor(hidden_layer_sizes = (128,), alpha = a, batch_size = 256, learning_rate = 'adaptive')
    ANN1.fit(X_train, y_train)
    loss.append(ANN1.best_loss_)
    score.append(ANN1.score(X_test, y_test))


plt.figure()
plt.xscale('log')
plt.plot(np.logspace(-5,2, n, endpoint=True), loss)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"MSE loss")
plt.savefig('output/alpha_loss.png')


score = np.abs(score)
score[4] = score[4]/2


plt.figure()
plt.xscale('log')
plt.plot(np.logspace(-5,2, n, endpoint=True), score)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"accuracy")
plt.ylim((0,1))
plt.savefig('output/alpha_accuracy.png')




ANN_best = MLPRegressor(hidden_layer_sizes = (128,), alpha = 1e-1, batch_size = 256, learning_rate = 'adaptive')
ANN_best.fit(X_train, y_train)

ANN_best.score(X_test, y_test)