"""
ECE 681 Homework 02 K-Nearest Neighbors and Bias-Variance Tradeoff

This file contains the functions and code to generate ROCS

Author: Ying Xu
NetID: yx136
Date: 02/10/2019
"""
import numpy as np

from sklearn import neighbors

from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt

from sklearn import metrics


"""
Read CSV file to get features
"""
y_label = np.loadtxt('dataSetHorseshoes.csv', delimiter=',', dtype='int', usecols=(0,))
X = np.loadtxt('dataSetHorseshoes.csv', delimiter=',', dtype='float', usecols=(1,2))
Test = np.loadtxt('dataSetHorseshoesTest.csv', delimiter=',', dtype='float', usecols=(1,2))
Test_label = np.loadtxt('dataSetHorseshoesTest.csv', delimiter=',', dtype='int', usecols=(0,))
# set classifier and fit
clf1 = neighbors.KNeighborsClassifier(n_neighbors = 1)
clf1.fit(X, y_label)
Z1 = clf1.predict_proba(Test)
FalseA1, Detect1, Thresholds1 = metrics.roc_curve(Test_label, Z1[:, 1])
max1 = 0
for i in range(0, len(FalseA1)):
    pcd1 = 0.5 * (1-FalseA1[i]) + 0.5 * Detect1[i]
    if pcd1 > max1:
        max1 = pcd1
plt.plot(FalseA1, Detect1, '-', color='darkorange', label='ROC of K = %d, maxPcd = %.2f' % (1, max1))


clf5 = neighbors.KNeighborsClassifier(n_neighbors = 5)
clf5.fit(X, y_label)
Z5 = clf5.predict_proba(Test)
FalseA5, Detect5, Thresholds5 = metrics.roc_curve(Test_label, Z5[:, 1])
max5 = 0
for i in range(0, len(FalseA5)):
    pcd5 = 0.5 * (1-FalseA5[i]) + 0.5 * Detect5[i]
    if pcd5 > max5:
        max5 = pcd5
plt.plot(FalseA5, Detect5, '-', color='green', label='ROC of K = %d, maxPcd = %.2f' % (5, max5))

clf31 = neighbors.KNeighborsClassifier(n_neighbors = 31)
clf31.fit(X, y_label)
Z31 = clf31.predict_proba(Test)
FalseA31, Detect31, Thresholds31 = metrics.roc_curve(Test_label, Z31[:, 1])
max31 = 0
for i in range(0, len(FalseA31)):
    pcd31 = 0.5 * (1-FalseA31[i]) + 0.5 * Detect31[i]
    if pcd31 > max31:
        max31 = pcd31
plt.plot(FalseA31, Detect31, '-', color='pink', label='ROC of K = %d, maxPcd = %.2f' % (31, max31))


clf91 = neighbors.KNeighborsClassifier(n_neighbors = 91)
clf91.fit(X, y_label)
Z91 = clf91.predict_proba(Test)
FalseA91, Detect91, Thresholds91 = metrics.roc_curve(Test_label, Z91[:, 1])
max91 = 0
for i in range(0, len(FalseA91)):
    pcd91 = 0.5 * (1-FalseA91[i]) + 0.5 * Detect91[i]
    if pcd91 > max91:
        max91 = pcd91
plt.plot(FalseA91, Detect91, '-', color='red', label='ROC of K = %d, maxPcd = %.2f' % (91, max91))

# plot title and axis
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('PFA')
plt.ylabel('PD')
plt.title('ROC')

plt.legend(loc="lower right")

plt.show()