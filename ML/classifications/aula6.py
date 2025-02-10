from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold, GridSearchCV
import pandas as pd
import   numpy as np

if "__main__" == __name__:

    pd.set_option('display.max_columns', 64)
    pd.set_option('display.max_rows', 64)
    arquivo = pd.read_csv('archive5/Data_train_reduced.csv')

    arquivo.drop('Product', axis=1, inplace=True)
    arquivo.drop('q1_1.personal.opinion.of.this.Deodorant', axis=1, inplace=True)

    arquivo.drop("q8.2", axis=1, inplace=True)
    arquivo.drop("q8.8", axis=1, inplace=True)
    arquivo.drop("q8.9", axis=1, inplace=True)
    arquivo.drop("q8.10", axis=1, inplace=True)
    arquivo.drop("q8.17", axis=1, inplace=True)
    arquivo.drop("q8.18", axis=1, inplace=True)
    arquivo.drop("q8.20", axis=1, inplace=True)

    arquivo["q8.12"].fillna(arquivo["q8.12"].mean(), inplace=True)
    arquivo["q8.7"].fillna(arquivo["q8.12"].mean(), inplace=True)

    y = arquivo["Instant.Liking"]
    x = arquivo.drop('Instant.Liking', axis=1)

    # Defina grupos, por exemplo, a partir de uma coluna do dataset
    groups = arquivo['Product.ID']  # Substitua 'GroupColumn' pelo nome correto da coluna de grupos

    #Definindo os parametros  que serão variados
    valores_C = np.array([0.01, 0.1, 0.5,  1,  2, 3, 5,  10, 20, 50, 100])
    regularizacao  = ['l1', 'l2']
    valores_grid  = {'C':valores_C, 'penalty':regularizacao}

    # kfold = StratifiedGroupKFold(n_splits=5)
    modelo = LogisticRegression(max_iter=1000)


    grid_regressao_logistica = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5)
    grid_regressao_logistica.fit(x,y)
    print(f"melhor   Acuracia: {grid_regressao_logistica.best_score_}")
    print(f"Parametro C: {grid_regressao_logistica.best_estimator_.C}")
    print(f"Regularizacao: {grid_regressao_logistica.best_estimator_.penalty}")

    # Passe os grupos no cross_val_score
    resultado = cross_val_score(modelo, x, y, cv=kfold, groups=groups)
    print(resultado.mean())
