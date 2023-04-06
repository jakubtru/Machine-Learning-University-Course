"""
Posortuj oba zbiory tak, aby rekordy były uporządkowane według cyfry którą reprezentują. Można
to zrobić na przykład tak:
1. Dla wygody utwórz osobne DataFrame’y z cechami i etykietami (nazwij je na przykład X i y).
2. Posortuj zbiór y rosnąco. Zauważ, że zmieniła się kolejność elementów w serii zbiorze, ale
każdy element „powędrował” wraz ze swoim indeksem (obejrzyj y.index).
3. Posortuj identycznie zbiór X wykorzystując funkcję reindex.
Podziel ręcznie posortowane dane na zbiory uczący i testujący w proporcjach 80-20 (pierwsze 80%–
następne 20%).
"""

"""
Naucz klasyfikator Stochastic Gradient Descent wykrywać cyfrę 0.
Policz dokładność ww. klasyfikatora na zbiorze uczącym oraz na zbiorze testującym.
Zapisz wyniki jako listę (list(float)) w pliku Pickle o nazwie sgd_acc.pkl.

Policz 3-punktową walidację krzyżową dokładności (accuracy) modelu dla zbioru uczącego. Zapisz
wynik jako tablicę (ndarray(3,)) w pliku Pickle o nazwie sgd_cva.pkl.
"""

"""
Utwórz macierz błędów dla zbioru testującego i zapisz ją jako tablicę (ndarray(10, 10)) w pliku
Pickle o nazwie sgd_cmx.pkl.
"""

import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist["data"], mnist["target"].astype(np.uint8)
print(X.shape,y.shape)

X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)

start = time.time()
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)

y_train_prediction = sgd_clf.predict(X_train)
train_accuracy = accuracy_score(y_train_0, y_train_prediction)

y_test_prediction = sgd_clf.predict(X_test)
test_accuracy = accuracy_score(y_test_0, y_test_prediction)

results_list = [train_accuracy, test_accuracy]

with open('sgd_acc.pkl', 'wb') as file:
    pickle.dump(results_list, file)

file.close()

score = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy", n_jobs=-1)

with open('sgd_cva.pkl', 'wb') as file:
    pickle.dump(score, file)

file.close()

sgd_m_clf = SGDClassifier(random_state=42, n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)
y_train_pred = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)
conf_matrix = confusion_matrix(y_train, y_train_pred)

with open('sgd_cmx.pkl', 'wb') as file:
    pickle.dump(conf_matrix, file)

file.close()
