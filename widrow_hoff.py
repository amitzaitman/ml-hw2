from math import sqrt
import numpy as np
import sklearn
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris, load_digits
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression


def plot_blobs(n_samples=(10, 500),
               center1_x=1.5,
               center1_y=1.5,
               center2_x=-1.5,
               center2_y=-1.5):
    centers = np.array([[center1_x, center1_y], [center2_x, center2_y]])
    global X, y
    X, y = make_blobs(n_samples=n_samples, n_features=2,
                      centers=centers, cluster_std=1.0)
    y = y * 2 - 1  # To convert to {-1, 1}

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='none')
    plt.xlim([-120, 120])
    plt.ylim([-120, 120])
    plt.grid()
    plt.axes().set_aspect('equal')



class widrow_hoff(object):
    def __init__(self, eta = .0041, tol = 1e-3):
        self.eta = eta
        self.tol = tol

    def fit(self, X, y):

        sc = StandardScaler()
        sc.fit(X)
        X = sc.transform(X)


        iterations = 0
        comp = 100

        rgen = np.random.RandomState()
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.costs = [0]

        while comp > self.tol:
            iterations = iterations + 1
            output = np.dot(X, self.w[1:]) + self.w[0]
            error = y - output
            self.w[1:] += self.eta * X.T.dot(error)
            self.w[0] += self.eta * error.sum()
            cost =  (error ** 2).sum() / 2.0
            comp = abs(cost - self.costs[-1])
            self.costs.append(cost)

        return iterations


    def predict(self, X):
        return np.where((np.dot(X, self.w[1:]) + self.w[0]) >= 0.0, 1, -1)





X_iris, y_iris = load_iris(return_X_y=True)
X_digits, y_digits = load_digits(return_X_y=True)


wh = widrow_hoff(tol=1e-3)
wh_iter_num_iris = wh.fit(X_iris, y_iris)
wh_iter_num_digits = wh.fit(X_digits, y_digits)

perc = Perceptron(tol=1e-3)
perc_iter_num_iris = perc.fit(X_iris, y_iris).n_iter_
perc_iter_num_digits = perc.fit(X_digits, y_digits).n_iter_

data = np.array([[wh_iter_num_iris, wh_iter_num_digits],
                 [perc_iter_num_iris, perc_iter_num_digits]])

df = pd.DataFrame(data, ["widrow_hoff", "Perceptron"], ["iris", "digits"])

print(df)

df.to_csv("iris.csv")

X = y = None  # Global variables


plt.figure()
plot_blobs(n_samples=(10, 500),
               center1_x=0.1,
               center1_y=0.1,
               center2_x=0,
               center2_y=0)
plt.figure()
perc_iter_blobs_1 = perc.fit(X, y).n_iter_
wh_iter_blobs_1 = wh.fit(X, y)

plot_blobs(n_samples=(10, 500),
               center1_x=100,
               center1_y=100,
               center2_x=-100,
               center2_y=-100)

perc_iter_blobs_2 = perc.fit(X, y).n_iter_
wh_iter_blobs_2 = wh.fit(X, y)

data = np.array([[wh_iter_blobs_1, wh_iter_blobs_2],
                 [perc_iter_blobs_1, perc_iter_blobs_2]])

df = pd.DataFrame(data, ["widrow_hoff", "Perceptron"], ["blob1", "blob2"])
df.to_csv("blob.csv")
print(df)

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20

classifiers = [
    ("SGD", SGDClassifier(max_iter=100, tol=1e-3)),
    ("ASGD", SGDClassifier(average=True, max_iter=1000, tol=1e-3)),
    ("Perceptron", Perceptron(tol=1e-3)),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0, tol=1e-4)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0, tol=1e-4)),
    ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0],
                               multi_class='auto')),
    ("widrow hoff", widrow_hoff(tol=1e-3))
]

xx = 1. - np.array(heldout)
plt.figure()

for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X_iris, y_iris, test_size=i, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()