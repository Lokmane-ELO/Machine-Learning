import numpy as np
from numpy.random import rand
import pylab as pl

def generateData(n):
    """
    generates a 2D linearly separable dataset with 2n samples.
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

X, Y = generateData(100)

pl.scatter(X[:, 0], X[:, 1],c=Y)


def k1(x1, x2):
	return 1 + x1[0]*x2[0] + x1[1]*x2[1] + x1[0]*x2[0]*x1[0]*x2[0] + x1[0]*x1[1]*x2[0]*x2[1] + x1[1]*x2[1]*x1[1]*x2[1]

def f_from_k(coeffs, support_set, k, x):
  f = 0
  for s, c in zip(support_set, coeffs):
    xx, y = s
    f += c * y * k(xx, x)
  return f

# perceptron noyau

dim = X.shape[1] 
w = np.zeros((dim,))
i = 0
coef = []
support = []
while True:
  error_count = 0
  for x, y in zip(X,Y):
    if y * f_from_k(coef, support, k1, x) <= 0: 
      if x not in support: 
        support.append((x,y))
        coef.append(1)
      else:
        coef[support.index((x,y))] += 1
      error_count += 1
  print(error_count)
  if error_count == 0:
    break

print(support, coef)

res = 500
for x in range(res):
    for y in range(res):
        if abs(f_from_k(coef, support, k1, [-3/2+3*x/res,-3/2+3*y/res])) < 0.01:
            pl.plot(-3/2+3*x/res,-3/2+3*y/res,'xr')

pl.title("données non linéairement séparables, plongement implicite par noyaux")
pl.show()

# reste à implémenter le noyau gaussien...
