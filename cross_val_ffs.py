from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#Extra-extra
from rfs import FFS

n_features = 100
n_samples = 5000
X, y = make_friedman1(n_samples=n_samples, n_features=n_features, random_state=0)
linear = LinearRegression()
score = []
n_lst = np.arange(1,20,1)
# for i in n_lst:
#     selector = RFE(linear, i, step=1, verbose=0)
#     selector.fit(X, y)
#     score.append(selector.score(X, y))
#
# plt.plot(n_lst, score, label="score")
# #plt.plot(test_sizes, test_error, label="test")
# plt.legend()
# plt.xlabel('number of features selected')
# plt.ylabel('R^2')
# plt.show()

for i in n_lst:
    selector = FFS(linear, i, step=1, verbose=0)
    selector.fit(X, y)
    score.append(selector.score(X, y))

plt.plot(n_lst, score, label="score")
#plt.plot(test_sizes, test_error, label="test")
plt.legend()
plt.xlabel('number of features selected')
plt.ylabel('R^2')
plt.show()
