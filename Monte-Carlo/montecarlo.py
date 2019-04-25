import numpy as np
import random

"""
P = #penny
Q = #quarter
D = dimension
"""
P = 5
Q = 5
D = 5


N= 10000
s = 0
for i in range(0, N):
    seprate_num = []
    penny_result = np.random.randint(2, size=[P, D])
    quarter_result = np.random.randint(2, size=[Q, D])

    the_same_flag = True
    for p in range(0, P):
        a = np.array(penny_result[p])
        for q in range(0, Q):
            b = np.array(quarter_result[q])
            seprate_num.append((a == b).all())
    seprate_num = np.array(seprate_num)
    if seprate_num.sum() == 0:
        s = s + 1

print(s/N)
