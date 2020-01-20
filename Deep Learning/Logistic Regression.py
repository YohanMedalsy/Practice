import numpy as np
import matplotlib.pylab as plt

np.random.seed(12)
num_observation = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observation)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75], [0.75, 1]], num_observation)
print(x1.shape)

x = np.vstack((x1, x2)).astype(np.float32)
print(np.zeros(num_observation).shape)
y = np.hstack((np.zeros(num_observation), np.ones(num_observation)))

plt.figure(figsize=(12, 4))
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.4)
plt.show()


def sigmoid(wx):
    return 1/(1 + np.exp(-wx))


def log_likelihod(x, y, w):
    h = sigmoid(np.dot(x, w.T))
    return np.dot(y.T, np.log(h)) + np.dot(1 - y.T, np.log(1 - h))


def logistic_regression(x, y, num_setps, learning_rate):
    intercept = np.ones((x.shape[0], 1))
    x = np.hstack((intercept, x))
    w = np.zeros(x.shape[1])
 
    for step in range(num_setps):
        wx = np.dot(x, w.T)
        est_y = sigmoid(wx)
        err = y - est_y
        gradient = np.dot(x.T, err)
        w = w + gradient * learning_rate

        if step % 10000 == 0:
            print(log_likelihod(x, y, w))

    return w


weights = logistic_regression(x, y, num_setps=100000, learning_rate=5e-5)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x, y)

print(clf.intercept_, clf.coef_)
print(weights)

data_with_intercept = np.hstack((np.ones((x.shape[0], 1)), x))
final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))

print('Accuracy from scratch: {0}'.format((preds == y).sum().astype(float) / len(preds)))
print('Accuracy from sk-learn: {0}'.format(clf.score(x, y)))

plt.figure(figsize=(12, 8))
plt.scatter(x[:, 0], x[:, 1],
            c=(preds == y) - 1, alpha=.8, s=50)
plt.show()





