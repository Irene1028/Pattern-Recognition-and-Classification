"""
ECE 681 Homework 02 K-Nearest Neighbors and Bias-Variance Tradeoff

This file contains the functions and code to generate
minPe for training and testing in Question4(e)

Author: Ying Xu
NetID: yx136
Date: 02/10/2019
"""
import numpy as np

from sklearn import neighbors

from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt

from sklearn import metrics

X_label = np.loadtxt('dataSetHorseshoes.csv', delimiter=',', dtype='int', usecols=(0,))
X = np.loadtxt('dataSetHorseshoes.csv', delimiter=',', dtype='float', usecols=(1,2))
Test = np.loadtxt('dataSetHorseshoesTest.csv', delimiter=',', dtype='float', usecols=(1,2))
Test_label = np.loadtxt('dataSetHorseshoesTest.csv', delimiter=',', dtype='int', usecols=(0,))

train_pe=[]
test_pe=[]

for i in range(1, 400):

    clf1 = neighbors.KNeighborsClassifier(n_neighbors = i)
    clf1.fit(X, X_label)
    Z_test = clf1.predict_proba(Test)
    Z_train = clf1.predict_proba(X)
    FalseA1, Detect1, Thresholds1 = metrics.roc_curve(X_label, Z_train[:, 1])
    max1 = 0
    for i in range(0, len(FalseA1)):
        pcd1 = 0.5 * (1-FalseA1[i]) + 0.5 * Detect1[i]
        if pcd1 > max1:
            max1 = pcd1
    train_pe.append(1-max1)

    FalseA2, Detect2, Thresholds2 = metrics.roc_curve(Test_label, Z_test[:, 1])
    max2 = 0
    for i in range(0, len(FalseA2)):
        pcd2 = 0.5 * (1 - FalseA2[i]) + 0.5 * Detect2[i]
        if pcd2 > max2:
            max2 = pcd2
    test_pe.append(1-max2)

print(train_pe)
k = [400/i for i in range(1, 400)]

plt.plot(k, train_pe, '-', color='blue', label='Training Data')
plt.plot(k, test_pe, '-', color='green', label='Testing Data')
# plot title and axis
plt.xlabel('N/k')
plt.ylabel('PE')

plt.legend(loc="lower right")

plt.show()