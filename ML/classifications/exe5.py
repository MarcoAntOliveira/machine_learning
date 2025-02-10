from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold, GridSearchCV
import pandas as pd  
import numpy as np

# importando o banco de dados do sklearn , e transformando em um dataframe
pd.set_option("display.max_columns", 30)
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
y = pd.Series(dados.target)
print(x.head())
# print(y.head(30))

# criando a lista com os parametros e os metodos de regularização 
valores_C = np.array([0.01, 0.1, 0.5,  1,  2, 3, 5,  10, 20, 50, 100])
valores_C = np.array([45, 47, 49, 50, 52, 53, 55, 56, 57, 58, 59])

# regularizacao  = ['l1', 'l2']
valores_grid  = {'C':valores_C}

# aplicando na regressão logistica
modelo = LogisticRegression(max_iter=1000)
grid_regressao_logistica = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5)
grid_regressao_logistica.fit(x,y)
print(f"melhor   Acuracia: {grid_regressao_logistica.best_score_}")
print(f"Parametro C: {grid_regressao_logistica.best_estimator_.C}")
# print(f"Regularizacao: {grid_regressao_logistica.best_estimator_.penalty}")