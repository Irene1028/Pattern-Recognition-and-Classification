"""
ECE 681 Homework 02 K-Nearest Neighbors and Bias-Variance Tradeoff

For Question5(b,c)
This file contains the code allowing us operate on different Pd = 0.95


Author: Ying Xu
NetID: yx136
Date: 02/10/2019
"""
import numpy as np

import random

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics
"""
Read CSV file to get features
"""
X_label = np.loadtxt('knn3DecisionStatistics.csv', delimiter=',', dtype='int', usecols=(0,))
X_scores = np.loadtxt('knn3DecisionStatistics.csv', delimiter=',', dtype='float', usecols=(1,))



"""calculate PFAs and PDs"""
N0 = 0
N1 = 0
for i in range(0, len(X_scores)):
    if X_label[i] == 0:
        N0 = N0+1
    else:
        N1 = N1+1
print(N0)
print(N1)

Pd_list = []
Pfa_list = []
for i in range(0, 1000):
    PD = 0
    PFA = 0
    for i in range(0, len(X_scores)):
        if X_label[i]==1:
            if X_scores[i] >= 0.3:
                PD = PD + 1
            else:
                rand_num = random.randint(1,4)
                if rand_num >= 2:
                    PD = PD + 1
        if X_label[i]==0:
            if X_scores[i] >= 0.3:
                PFA = PFA + 1
            else:
                rand_num = random.randint(1, 7)
                if rand_num >= 3:
                    PFA = PFA + 1
    PFA = PFA / N0
    PD = PD / N1
    Pd_list.append(PD)
    Pfa_list.append(PFA)

print(np.mean(Pd_list))

pfa, pd, thres = metrics.roc_curve(X_label, X_scores)
plt.plot(pfa, pd, 'o-', color='blue', label='ROC of K = 3')

plt.plot(np.mean(Pfa_list), np.mean(Pd_list),'ro', label = 'Pd(%.4f, %.4f)' % (np.mean(Pfa_list), np.mean(Pd_list)))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('PFA')
plt.ylabel('PD')
plt.title('ROC')

plt.legend(loc="lower right")

#plt.title("kernel density estimation")
#sns.distplot(Pd_list, rug= True, hist= False)

plt.show()
