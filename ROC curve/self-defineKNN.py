"""
ECE 681 Homework 02 K-Nearest Neighbors and Bias-Variance Tradeoff

This file contains the functions and code to generate
different KNNs in Question2

Author: Ying Xu
NetID: yx136
Date: 02/10/2019
"""

from math import pow
from collections import defaultdict
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


"""
    Neighbor class
    Attributes:
        class_label: truth of this neighbor
        distance: distance between test point and the neighbor
"""
class Neighbor(object):

    def __init__(self, class_label, distance):
        self.class_label = class_label
        self.distance = distance

"""
Self-defined KNN
"""
class KNeighborClassifier(object):

    def __init__(self, n_neighbors=5, metric='euclidean'):
        """
        initialized function
        n_neighbors: k value
        metric: L2, euclidean.
        """
        self.n_neighbors = n_neighbors
        if metric == 'euclidean':
            self.p = 2
        elif metric == 'manhattan':
            self.p = 1

    def fit(self, train_x, train_y):
        """
        fit function
        """
        self.train_x = train_x.astype(np.float32)
        self.train_y = train_y

    def predict_one(self, one_test):
        '''
        predict one single test point
        '''
        neighbors = []
        for x, y in zip(self.train_x, self.train_y):
            distance = self.get_distance(x, one_test)
            neighbors.append(Neighbor(y, distance))

        neighbors.sort(key=lambda x: x.distance)

        if self.n_neighbors > len(self.train_x):
            self.n_neighbors = len(self.train_x)

        cls_count = defaultdict(int)
        cls_count[0].__init__(0)
        cls_count[1].__init__(0)
        for i in range(self.n_neighbors):
            cls_count[neighbors[i].class_label] += 1
        #print(cls_count)
        t_label = max(cls_count, key=cls_count.get)
        lamda = cls_count[1]/self.n_neighbors
        #ans = np.c_[t_label, lamda]
        #print(ans)
        return t_label, lamda

    def predict(self, test_x):
        '''
        predict a testing list
        '''
        x1=[]
        x2=[]
        for x in test_x:
            test_truth, test_value = self.predict_one(x)
            x1.append(test_truth)
            x2.append(test_value)
        return x1, x2

    def get_distance(self, input, x):
        """
        calculate the distance between two points
        """
        if self.p == 2:
            return np.linalg.norm(input - x)#计算矩阵范数
        ans = 0
        for i, t in zip(input, x):
            ans += pow(abs(i - t), self.p)
        return pow(ans, 1 / self.p)


"""
Read CSV file to get features
"""
y_label = np.loadtxt('dataSetHorseshoes.csv', delimiter=',', dtype='int', usecols=(0,))
X = np.loadtxt('dataSetHorseshoes.csv', delimiter=',', dtype='float', usecols=(1, 2))

# set classifier and fit
clf = KNeighborClassifier(n_neighbors = 31)
clf.fit(X, y_label)

# set step
h = .02

# define max and min
f1x_min, f1x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
f2y_min, f2y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

# generate grid
xx, yy = np.meshgrid(np.arange(f1x_min, f1x_max, h),
                     np.arange(f2y_min, f2y_max, h))
Z_label, Z_proba = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z_proba)
Z_proba = np.array(Z_proba)
Z_proba = Z_proba.reshape(xx.shape)
print(Z_proba)

# Create color maps
cmap_bold = ListedColormap(['#0000CC', '#FF0000'])

# different color for different class
cm = plt.cm.get_cmap('cool')
background = plt.contourf(xx, yy, Z_proba, 31, cmap=cm, vmax= 1.0, vmin = 0)
plt.contour(xx, yy, Z_proba, [16/31], colors='yellow', linewidth = .2)

# plot training data
plt.scatter(X[:, 0], X[:, 1], c=y_label, marker='.', cmap=cmap_bold)

# plot title and axis
plt.title('K = 31')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.colorbar(background)
plt.show()