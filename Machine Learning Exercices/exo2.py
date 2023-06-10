import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC

data = np.loadtxt("./data.csv")
X = data[:, :-2]
Y = data[:, -2:]
Yheat = Y[:, 0]
Ycool = Y[:, 1]

target = np.loadtxt('./labels', dtype=int)

section = 3 * len(Yheat) // 4  # indice de la section entre les données d'apprentissage et de test (75%)

target_train = target[:section]
Yheat_train = Yheat[:section].reshape(-1, 1)  # 75% des données

target_test = target[section:]
Yheat_test = Yheat[section:].reshape(-1, 1)  # 25% des données

# Algo des k-plus proches voisins (n = 15 comme dans le TP)
neigh = KNeighborsClassifier(n_neighbors=15)
neigh.fit(Yheat_train, target_train)
score_neigh = neigh.score(Yheat_test, target_test)
print("Le score de l'algo des k-plus proches voisins :", score_neigh)

#  Algo des arbres de décision
tree = tree.DecisionTreeClassifier()
tree.fit(Yheat_train, target_train)
score_tree = tree.score(Yheat_test, target_test)
print("Le score de l'algo des arbres de décision :", score_tree)

#  Algo SVM à noyau linéaire
k_lin = SVC(kernel='linear')
k_lin.fit(Yheat_train, target_train)
score_k_lin = k_lin.score(Yheat_test, target_test)
print("Le score de l'algo SVM à noyau linéaire :", score_k_lin)

#  Algo SVM à noyau rbf (gamma = 'scale' car default selon la documentation de sklearn)
rbf = SVC(gamma='scale', kernel='rbf')
rbf.fit(Yheat_train, target_train)
score_rbf = rbf.score(Yheat_test, target_test)
print("Le score de l'algo SVM à noyau rbf :", score_rbf)
