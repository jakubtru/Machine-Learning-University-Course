"""
1. Podziel zbiór danych na uczący i testujący w proporcjach 80/20.
2. Zbuduj modele klasyfikacji SVM dla średnich (mean) wartości cech area oraz smoothness;
stwórz dwa modele:
    - LinearSVC, z funkcją straty “hinge”,
    - LinearSVC, z funkcją straty “hinge”, po uprzednim automatycznym skalowaniu wartości cech.
3. Policz dokładność (accuracy) dla ww. klasyfikacji osobno na zbiorze uczącym i testującym, zapisz wartości na liście w kolejności: zbiór uczący bez skalowania, zbiór testujący bez skalowania, zbiór uczący ze skalowanie, zbiór testujący ze skalowaniem. Listę zapisz w pliku Pickle
bc_acc.pkl.
4. Czy skalowanie coś dało?
5. Ekperyment powtórz dla zbioru irysów; zbuduj model wykrywający, czy dany przypadek jest
gatunku Virginica na podstawie cech: długość i szerokość płatka.
6. Policz dokładność (accuracy) dla w/w klasyfikacji osobno na zbiorze uczącym i testującym, zapisz wartości na liście w kolejności: zbiór uczący bez skalowania, zbiór testujący bez skalowania, zbiór uczący ze skalowanie, zbiór testujący ze skalowaniem. W.w. listę zapisz w pliku
Pickle iris_acc.pkl.
"""

from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data_breast_cancer = datasets.load_breast_cancer()
print(data_breast_cancer['DESCR'])

data_iris = datasets.load_iris()
print(data_iris['DESCR'])

df = pd.DataFrame(data_breast_cancer.data, columns = data_breast_cancer.feature_names)
X_train, X_test, y_train, y_test = train_test_split(df[['mean area', 'mean smoothness']], data_breast_cancer.target, test_size=0.2, random_state=42)

svm1 = LinearSVC(loss="hinge")
svm1.fit(X_train, y_train)
y_pred_train1 = svm1.predict(X_train)
y_pred_test1 = svm1.predict(X_test)

X_train_mean_scaled = StandardScaler().fit_transform(X_train)
svm2 = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge",random_state=444))])
svm2.fit(X_train_mean_scaled, y_train)
y_pred_train2 = svm2.predict(X_train_mean_scaled)
y_pred_test2 = svm2.predict(StandardScaler().fit_transform(X_test))

lista = []
lista.append(accuracy_score(y_train, y_pred_train1))
lista.append(accuracy_score(y_test, y_pred_test1))
lista.append(accuracy_score(y_train, y_pred_train2))
lista.append(accuracy_score(y_test, y_pred_test2))

with open('bc_acc.pkl', 'wb') as file:
    pickle.dump(lista, file)

X = data_iris.data[:, 2:]
y = (data_iris.target == 2).astype(np.int8)

svm_clf1 = LinearSVC(loss='hinge', random_state=444)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=444)
svm_clf1.fit(X_train, y_train)

y_pred_train1 = svm_clf1.predict(X_train)
y_pred_test1 = svm_clf1.predict(X_test)

svm_clf2 = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge",random_state=444))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=444)
svm_clf2.fit(X_train, y_train)

y_pred_train2 = svm_clf2.predict(X_train)
y_pred_test2 = svm_clf2.predict(X_test)


lista2 = []
lista2.append(accuracy_score(y_train, y_pred_train1))
lista2.append(accuracy_score(y_test, y_pred_test1))
lista2.append(accuracy_score(y_train, y_pred_train2))
lista2.append(accuracy_score(y_test, y_pred_test2))

with open('iris_acc.pkl', 'wb') as file:
    pickle.dump(lista2, file)
