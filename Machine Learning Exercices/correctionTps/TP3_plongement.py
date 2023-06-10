import numpy as np
from numpy.random import rand
import pylab as pl

def generateData3(n):
    """
    generates a 2D non linearly separable dataset with 2n samples.
    The third element of the sample is the label
    """
    xb = (rand(n) * 2 - 1) / 2
    yb = (rand(n) * 2 - 1) / 2
    xr = 3 * (rand(4 * n) * 2 - 1) / 2
    yr = 3 * (rand(4 * n) * 2 - 1) / 2
    inputs = []
    for i in range(n):
        inputs.append([xb[i], yb[i], -1])

    for i in range(4 * n):
        if abs(xr[i]) >= 1 or abs(yr[i]) >= 1:
            inputs.append([xr[i], yr[i], 1])

    data = np.array(inputs)
    X = data[:, 0:2]
    Y = data[:, -1]
    return X, Y

X, Y = generateData3(100)


pl.scatter(X[:, 0], X[:, 1],c=Y, s=20)
#pl.show()

#stophere


# fonction de plongement explicite des X
def phi(X):
    XX = []
    for i in range(len(X)):
        x = X[i]
        #XX.append([1, x[0], x[1], x[0]*x[0], x[0] * x[1], x[1]*x[1]])
        XX.append([x[0], x[1], (x[0]*x[0])+(x[1]*x[1])-1]) #, x[0] * x[1], x[1]*x[1]])
    return np.array(XX)

# application du plongement sur les X
Xorig = X
X = phi(X)

fig = pl.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=Y)
#pl.show()

# perceptron classique
dim = X.shape[1]
w = np.zeros((dim,))
j = 0
error_count = 1
while error_count>0:
  error_count = 0
  for i in range(len(X)):
    if Y[i] * np.dot(X[i], w) <= 0:
      error_count += 1
      w = w + X[i]*Y[i]
  print(j, error_count)
  j = j + 1

print(w, j)


# evaluation de la fct f de la courbe separatrice
def f(w, x, y):
    return np.dot(phi([[x, y]]), w)

res = 500
for x in np.linspace(min(Xorig[:,0]), max(Xorig[:,0]), res):
    for y in np.linspace(min(Xorig[:,1]), max(Xorig[:,1]), res):
        #print(f(w,-3/2+3*x/res,-3/2+3*y/res))
        if abs(f(w,x,y)) < 0.01:
            pl.plot(x,y,'xr')

pl.title("données non linéairement séparables, fonction de plongement explicite")
pl.show()
