from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet
# import matplotlib.pyplot as plt
import numpy as np

x, y = make_regression(n_samples=200, n_features=1, noise=30)
x_treino , x_teste, y_treino, y_teste = train_test_split(x, y, test_size= 0.30)

modeloLinear  =  LinearRegression()
modeloLinear.fit(x_treino, y_treino)
resultadoLinear = modeloLinear.score(x_teste, y_teste)
print(f" O resultado do modelo regressao linear é  {resultadoLinear}")


modeloRidge = Ridge(alpha=10)
modeloRidge.fit(x_treino, y_treino)
resultadoRidge = modeloRidge.score(x_teste, y_teste)
print(f"O resultado do ridge regression é {resultadoRidge}")

modeloLasso = Lasso(alpha=10, max_iter=1000, tol= 0.01)
modeloLasso.fit(x_treino, y_treino)
resultadoLasso = modeloLasso.score(x_teste, y_teste)
print(f" O resultado do modelo de Lasso é  {resultadoLasso}")

modeloElasticNet = ElasticNet(alpha=10, l1_ratio=0.5, tol= 0.1, max_iter=10000)
modeloElasticNet.fit(x_treino, y_treino)
resultadoElasticnet = modeloElasticNet.score(x_teste, y_teste)
print(f" O resultado do modelo Elastic net é  {resultadoElasticnet}")