import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from sklearn.model_selection import train_test_split

print('')
print('Partie 2')
print('')

# chargement des donnees
data = np.loadtxt('dataRegLin2D.txt')
print("data.shape = ", data.shape)

X = data[:, :2]
Y = data[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print("X.shape = ", X_train.shape)
print("y.shape = ", Y_train.shape)

# plot 3D
fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# plot 2D
ax = fig.add_subplot(132)
ax.scatter(X[:,0], Y)

ax = fig.add_subplot(133)
ax.scatter(X[:,1], Y)

#plt.show()


#sys.exit(0)


# complement avec les 1
def complete(X):
    ones = np.ones((X.shape[0], 1))
    XX = np.concatenate((X, ones), axis=-1)
    return XX

# regression
def fit(X, Y):
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)
    #w = np.linalg.inv(X.T @ X) @ X.T @ Y
    return w

def predict(x, W):
	return np.dot(W[:W.shape[0]-1], x) + W[-1]
    #return np.dot(W, X)

def score(X, Y, W, idx):
  err = 0.0
  for i in range(len(X)):
    y = predict(X[i,:idx], W)
    #y = regr(X[i], W)
    err += (y-Y[i])*(y-Y[i])
  err /= len(X)
  return err


X1 = complete(X_train[:, 0].reshape((len(X_train), 1)))
W1 = fit(X1, Y_train)
X1 = complete(X_test[:, 0].reshape((len(X_test), 1)))
score1 = score(X1, Y_test, W1, 1)
print("Error1 = ", score1)
print("W1", W1)

X2 = complete(X_train[:, 1].reshape((len(X_train), 1)))
W2 = fit(X2, Y_train)
X2 = complete(X_test[:, 1].reshape((len(X_test), 1)))
score2 = score(X2, Y_test, W2, 1)
print("Error2 = ", score2)
print("W2", W2)

#y = mx + p
#vec directeur = (1, m)
#
#ax + by + c = 0
#vec directeur = (-b, a)

x1 = X2.min()
x2 = X2.max()
y1 = W2[0]*x1+W2[1]
y2 = W2[0]*x2+W2[1]
ax.plot([x1,x2],[y1,y2])



X3 = complete(X_train)
W3 = fit(X3, Y_train)
X3 = complete(X_test)
score3 = score(X3, Y_test, W3, 2)
print("Error3 = ", score3)
print("W3", W3)


plt.show()


#sys.exit(0)


########## partie 2 #########

print('')
print('Partie 2')
print('')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

X, Y = load_diabetes(return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

clf = LinearRegression().fit(X_train, Y_train)
Y_train_pred = clf.predict(X_train)
print('Erreur empirique - LinearRegression:', mean_squared_error(Y_train, Y_train_pred))
Y_test_pred = clf.predict(X_test)
print('Erreur généralisation - LinearRegression:', mean_squared_error(Y_test, Y_test_pred))
print('')

alphas = np.logspace(-3, -1, 20)
for model in [(Ridge, 'Ridge'), (Lasso, 'Lasso')]:
    gscv = GridSearchCV(model[0](), dict(alpha=alphas), cv=10, refit=True).fit(X_train, Y_train)
    Y_train_pred = gscv.predict(X_train)
    print('Erreur empirique - '+str(model[1])+':', mean_squared_error(Y_train, Y_train_pred))
    Y_test_pred = gscv.predict(X_test)
    print('Erreur généralisation - '+str(model[1])+':', mean_squared_error(Y_test, Y_test_pred))
    print('  W = ', gscv.best_estimator_.coef_)
    print('')


