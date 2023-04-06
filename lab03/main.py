"""
Wykonaj następujące regresje na ww. zbiorze:
1. liniową,
2. KNN, dla 𝑘 = 3 oraz 𝑘 = 5,
3. wielomianową 2, 3, 4 i 5 rzędu.

Przeanalizuj działanie każdej z otrzymanych funkcji regresyjnych. Porównaj ich przebiegi z
rozkładem zbioru danych.

Zapisz w osobnym DataFrame wartości MSE dla zbiorów uczących i testujących dla ww.
regresorów; kolumny: train_mse, test_mse, wiersze: lin_reg, knn_3_reg, knn_5_reg,
poly_2_reg, poly_3_reg, poly_4_reg, poly_5_reg. Zapisz ww. DataFrame do pliku Pickle
o nazwie: mse.pkl

Zapisz do pliku Pickle o nazwie reg.pkl listę krotek zawierających obiekty reprezentujące
regresory: [(lin_reg, None), (knn_3_reg, None), (knn_5_reg, None), (poly_2_reg,
poly_feature_2), (poly_3_reg, poly_feature_3), (poly_4_reg, poly_feature_4),
(poly_5_reg, poly_feature_5)]
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


size = 300
X = np.random.rand(size) * 5 - 2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4 * (X ** 4) + w3 * (X ** 3) + w2 * (X ** 2) + w1 * X + w0 + np.random.randn(size) * 8 - 4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv', index=None)
df.plot.scatter(x='x', y='y')

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set), len(test_set)
X_train = train_set[['x']]
y_train = train_set['y']
X_test = test_set[['x']]
y_test = test_set['y']


regression = LinearRegression()

regression.fit(X_train, y_train)

lin_pred = regression.predict(X_test)
lin_mse_test = mean_squared_error(y_test, lin_pred)

lin_pred_t = regression.predict(X_train)
lin_mse_train = mean_squared_error(y_train, lin_pred_t)

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.scatter(df['x'], df['y'], s=5)

plt.subplot(2, 2, 2)
plt.scatter(X_test, y_test, s=5, label='')
plt.scatter(X_train, y_train, s=5, label='')
plt.plot(X_test, lin_pred, color='red', linewidth=2, )

knn3 = KNeighborsRegressor(n_neighbors=3)
knn3.fit(X_train.values.reshape(-1, 1), y_train)

knn3_mse_test = mean_squared_error(y_test, knn3.predict(X_test))
knn3_mse_train = mean_squared_error(y_train, knn3.predict(X_train))
x_p = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
y_pred = w4 * (x_p ** 4) + w3 * (x_p ** 3) + w2 * (x_p ** 2) + w1 * x_p + w0

plt.figure(figsize=(6, 4))
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(x_p, y_pred)
plt.plot(x_p, knn3.predict(x_p))
plt.show()
print(knn3)

knn5 = KNeighborsRegressor(n_neighbors=5)
knn5.fit(X_train.values.reshape(-1, 1), y_train)

x_p = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
y_pred = w4 * (x_p ** 4) + w3 * (x_p ** 3) + w2 * (x_p ** 2) + w1 * x_p + w0

plt.figure(figsize=(6, 4))
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.plot(x_p, y_pred)
plt.plot(x_p, knn5.predict(x_p))
plt.show()

knn5_mse_test = mean_squared_error(y_test, knn5.predict(X_test))
print(knn5_mse_test)
knn5_mse_train = mean_squared_error(y_train, knn5.predict(X_train))
print(knn5_mse_train)
print(knn5)

poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly2.fit_transform(X_train)
X_test_poly = poly2.transform(X_test)

poly2_reg = LinearRegression()
poly2_reg.fit(X_train_poly, y_train)
poly2_pred = poly2_reg.predict(X_test_poly)

poly2_mse_test = mean_squared_error(y_test, poly2_reg.predict(X_test_poly))
poly2_mse_train = mean_squared_error(y_train, poly2_reg.predict(X_train_poly))

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)
X_pred = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
X_poly_pred = poly2.fit_transform(X_pred)
y_pred = lin_reg.predict(X_poly_pred)

plt.scatter(X_train, y_train)
plt.plot(X_pred, y_pred, color='red')
plt.show()

poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly3.fit_transform(X_train)
X_test_poly = poly3.transform(X_test)

poly3_reg = LinearRegression()
poly3_reg.fit(X_train_poly, y_train)
poly3_pred = poly3_reg.predict(X_test_poly)

poly3_mse_test = mean_squared_error(y_test, poly3_reg.predict(X_test_poly))
poly3_mse_train = mean_squared_error(y_train, poly3_reg.predict(X_train_poly))

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)
X_pred = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
X_poly_pred = poly3.fit_transform(X_pred)
y_pred = lin_reg.predict(X_poly_pred)

plt.scatter(X_train, y_train)
plt.plot(X_pred, y_pred, color='red')
plt.show()

poly4 = PolynomialFeatures(degree=4, include_bias=False)
X_train_poly = poly4.fit_transform(X_train)
X_test_poly = poly4.transform(X_test)

poly4_reg = LinearRegression()
poly4_reg.fit(X_train_poly, y_train)
poly4_pred = poly4_reg.predict(X_test_poly)

poly4_mse_test = mean_squared_error(y_test, poly4_reg.predict(X_test_poly))
poly4_mse_train = mean_squared_error(y_train, poly4_reg.predict(X_train_poly))

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)
X_pred = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
X_poly_pred = poly4.fit_transform(X_pred)
y_pred = lin_reg.predict(X_poly_pred)

plt.scatter(X_train, y_train)
plt.plot(X_pred, y_pred, color='red')
plt.show()

poly5 = PolynomialFeatures(degree=5, include_bias=False)
X_train_poly = poly5.fit_transform(X_train)
X_test_poly = poly5.transform(X_test)

poly5_reg = LinearRegression()
poly5_reg.fit(X_train_poly, y_train)
poly5_pred = poly5_reg.predict(X_test_poly)

poly5_mse_test = mean_squared_error(y_test, poly5_reg.predict(X_test_poly))
poly5_mse_train = mean_squared_error(y_train, poly5_reg.predict(X_train_poly))

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)
X_pred = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
X_poly_pred = poly5.fit_transform(X_pred)
y_pred = lin_reg.predict(X_poly_pred)

plt.scatter(X_train, y_train)
plt.plot(X_pred, y_pred, color='red')
plt.show()

data_mse = pd.DataFrame({'train_mse': [lin_mse_train, knn3_mse_train, knn5_mse_train, poly2_mse_train, poly3_mse_train,
                                       poly4_mse_train, poly5_mse_train],
                         'test_mse': [lin_mse_test, knn3_mse_test, knn5_mse_test, poly2_mse_test, poly3_mse_test,
                                      poly4_mse_test, poly5_mse_test]},
                        index=['lin_reg', 'knn_3_reg', 'knn_5_reg', 'poly_2_reg', 'poly_3_reg', 'poly_4_reg',
                               'poly_5_reg'])

with open('mse.pkl', 'wb') as file:
    pickle.dump(data_mse, file)

lista_krotek = [(regression, None), (knn3, None), (knn5, None), (poly2_reg, poly2), (poly3_reg, poly3),
                (poly4_reg, poly4), (poly5_reg, poly5)]

with open('reg.pkl', 'wb') as file:
    pickle.dump(lista_krotek, file)
