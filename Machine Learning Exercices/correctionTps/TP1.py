import sys
from sklearn.datasets import load_iris
import pylab as pl
from sklearn import neighbors
import numpy as np

irisData = load_iris()

X = irisData.data
Y = irisData.target

'''
for x in range(4):
  for y in range(x, 4):
    if x != y:
      pl.scatter(X[:, x], X[:, y],c=Y, s=100)
      pl.title(str(x)+' '+str(y))
      #pl.legend()
      pl.show()
'''

nb_voisins = 15
clf = neighbors.KNeighborsClassifier(nb_voisins)
clf.fit(X, Y)

#print("predict", clf.predict([[ 5.4,  3.2,  1.6,  0.4]]))
#print("predict_proba", clf.predict_proba([[ 5.4,  3.2,  1.6,  0.4]]))

print('accuracy all', clf.score(X,Y))

#print("predict complet")
#Z = clf.predict(X)
#print("X[Z!=Y]")
#print(X[Z!=Y])



from sklearn.model_selection import train_test_split, KFold
import random # pour pouvoir utiliser un g ́en ́erateur de nombres al ́eatoires 

### question 2.2

score_train = []
score_test = []
for i in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random.seed(123))

    clf.fit(X_train, Y_train)
    score_train.append(clf.score(X_train,Y_train))
    score_test.append(clf.score(X_test,Y_test))

print("accuracy train", np.mean(score_train))
print("accuracy test", np.mean(score_test))


#sys.exit()
### question 2.3

score_train = []
score_test = []

scores_all = [0] * 30 
for i in range(10): # repeat multiple splits
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=random.seed(123))

    kf = KFold(n_splits=10,shuffle=True)
    scores=[]
    for k in range(1,30):
      score=0
      clf = neighbors.KNeighborsClassifier(k)
      for learn,test in kf.split(X_train):
        X_train2 = X_train[learn]
        Y_train2 = Y_train[learn]
        clf.fit(X_train2, Y_train2)
        X_test2 = X_train[test]
        Y_test2 = Y_train[test]
        score = score + clf.score(X_test2, Y_test2)
      scores_all[k-1] += score
    print(i)
k_opt = scores_all.index(max(scores_all))+1
print("meilleure valeur pour k : ", k_opt)

clf = neighbors.KNeighborsClassifier(k_opt)
clf.fit(X_train, Y_train) #refit with optimal parameter

score_train_best = clf.score(X_train,Y_train)
score_test_best = clf.score(X_test,Y_test)


print("accuracy train", score_train_best)
print("accuracy test", score_test_best)

