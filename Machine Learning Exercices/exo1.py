import numpy as np
from sklearn.linear_model import LinearRegression

data = np.loadtxt("./data.csv")
X = data[:, :-2]
Y = data[:, -2:]
Yheat = Y[:, 0]
Ycool = Y[:, 1]

section = 3 * len(Yheat) // 4  # indice de la section entre les données d'apprentissage et de test

Yheat_train = Yheat[:section].reshape(-1, 1)
X_train = X[:section]  # 75% des données

Yheat_test = Yheat[section:].reshape(-1, 1)
X_test = X[section:]  # 25% des données

reg = LinearRegression()

reg.fit(X_train, Yheat_train)
score = reg.score(X_test, Yheat_test)
print("Le score sans supprimer d'attribut :", score)

meilleur_score = [0, -10]
for i in np.arange(0, X_train.shape[1]):
    reg.fit(np.delete(X_train, i, axis=1), Yheat_train)
    score = reg.score(np.delete(X_test, i, axis=1), Yheat_test)
    if score > meilleur_score[1]:
        meilleur_score = [i, score]

print("C'est sans l'attribut", meilleur_score[0]+1, "qu'on a le meilleur score : ", meilleur_score[1])
