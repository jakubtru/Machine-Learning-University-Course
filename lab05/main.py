from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import *
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pickle

""" Klasyfikacja
1. Użyj drzew decyzyjnych do klasyfikacji zbioru danych data_breast_cancer dla cech mean
texture i mean symmetry.
2. Podziel ww. zbiór na uczący i testujący w proprcjach 80:20.
3. Znajdź odpowiednią głębokośc drzewa decyzyjnego, tak aby osiągnąć maksymalną wartość f1.
4. Wygeneruj rysunek drzewa decyzyjnego w pliku bc.png.
5. Zapisz w pliku Pickle f1acc_tree.pkl listę zawierającą: głebokość drzewa, f1 dla zbioru
uczącego, f1 dla zbioru testowego, dokładność (accuracy) dla zbioru uczącego, dokładność
(accuracy) dla zbioru testowego.
"""

data_breast_cancer = datasets.load_breast_cancer(as_frame=True)

df = pd.DataFrame(data_breast_cancer.data, columns=data_breast_cancer.feature_names)
X_train, X_test, y_train, y_test = train_test_split(df[['mean texture', 'mean symmetry']], data_breast_cancer.target,
                                                    test_size=0.2, random_state=42)

max_train = 0
max_test = 0
optimal_depth_train = 0
optimal_depth_test = 0

for i in range(1, 10):
    tree_clf = DecisionTreeClassifier(max_depth=i, random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_train = tree_clf.predict(X_train)
    y_pred_test = tree_clf.predict(X_test)

    f1_train = f1_score(y_true=y_train, y_pred=y_pred_train)
    f1_test = f1_score(y_true=y_test, y_pred=y_pred_test)

    if f1_train > max_train:
        max_train = f1_train
        optimal_depth_train = i
    if f1_test > max_test:
        max_test = f1_test
        optimal_depth_test = i

tree_clf = DecisionTreeClassifier(max_depth=optimal_depth_test, random_state=42)
tree_clf.fit(X_train, y_train)

f = "bc.png"
export_graphviz(tree_clf, out_file=f,
                feature_names=data_breast_cancer.feature_names[28:],
                class_names=[str(num) + ", " + name
                             for num, name in
                             zip(set(data_breast_cancer.target), data_breast_cancer.target_names)],
                rounded=True, filled=True)

y_pred_train = tree_clf.predict(X_train)
y_pred_test = tree_clf.predict(X_test)
f1_train = f1_score(y_true=y_train, y_pred=y_pred_train)
f1_test = f1_score(y_true=y_test, y_pred=y_pred_test)
results = [optimal_depth_test, f1_train, f1_test, accuracy_score(y_train, y_pred_train),
           accuracy_score(y_test, y_pred_test)]

with open('f1acc_tree.pkl', 'wb') as file:
    pickle.dump(results, file)

""" Regresja
1. Użyj drzew decyzyjnych do budowy regresora na zbiorze danych df.
2. Podziel w/w zbiór na uczący i testujący w proprcjach 80/20.
3. Znajdź odpowiednią głębokośc drzewa decyzyjnego, tak aby wartość błędu średniokwadratowego (MSE) były jak najmniejsze (uwaga na
overfitting).
4. Sporządź wykres wszystkich danych z df oraz predykcji regresora, porównaj wyniki z tymi
osiągniętymi dla regresji wielomianowej i KNN z poprzednich ćwiczeń.
5. Wygenruj rysunek drzewa decyzyjnego w pliku reg.png.
6. Zapisz w pliku Pickle mse_tree.pkl listę zawierającą: głebokość drzewa, MSE dla zbioru
uczącego, MSE dla zbioru testowego.
"""

size = 300
X = np.random.rand(size) * 5 - 2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4 * (X ** 4) + w3 * (X ** 3) + w2 * (X ** 2) + w1 * X + w0 + np.random.randn(size) * 8 - 4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x', y='y')

X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)

min_test = 100
min_train = 100
optimal_depth_test = 0
optimal_depth_train = 0
for i in range(1, 10):
    tree_reg = DecisionTreeRegressor(max_depth=i, random_state=42)
    tree_reg.fit(X_train, y_train)

    y_pred_test = tree_reg.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)

    y_pred_train = tree_reg.predict(X_train)
    mse_train = mean_squared_error(y_train, y_pred_train)

    if mse_test < min_test:
        min_test = mse_test
        optimal_depth_test = i
    if mse_train < min_train:
        min_train = mse_train
        optimal_depth_train = i

tree_reg = DecisionTreeRegressor(max_depth=optimal_depth_test, random_state=42)
tree_reg.fit(X_train, y_train)

f = "reg.png"
export_graphviz(tree_reg, out_file=f,
                rounded=True, filled=True)

y_pred_test = tree_reg.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
y_pred_train = tree_reg.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
results = [optimal_depth_test, mse_train, mse_test]

with open('mse_tree.pkl', 'wb') as file:
    pickle.dump(results, file)
