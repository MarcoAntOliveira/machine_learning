

from functions import modelosRegressao, modelosRegressaoKfold
from sklearn.datasets import make_regression

import pandas as pd


if "__main__" == __name__:
  # x, y = make_regression(n_samples=200, n_features=1, noise=30)
  # modelosRegressao(x, y)

  # pd.set_option("display.max_columns", 42)
  # dados = pd.read_csv("kc_house_data.csv")

  # dados.drop('id', axis=1, inplace=True)
  # dados.drop('date', axis=1, inplace=True)
  # dados.drop('zipcode', axis=1, inplace=True)
  # dados.drop('lat', axis=1, inplace=True)
  # dados.drop('long', axis=1, inplace=True)

  # y = dados["price"]
  # x = dados.drop('price', axis=1)

  colunas = ['No' , 'gre' , 'toefl', 'rating', 'sop', 'lor', 'cgpa', 'research', 'chance']
  pd.set_option("display.max_columns", 320)
  dados = pd.read_csv("archive4/Admission_Predict_Ver1.1.csv", names = colunas)
  # dados = pd.read_csv("kc_house_data.csv")
  dados = dados.drop(0)
  dados.drop('No', axis=1, inplace=True)

  y = dados["chance"]
  x = dados.drop('chance', axis=1)

  test_size = 0.4
  modelosRegressaoKfold(x, y, test_size)

  test_size = 0.5
  modelosRegressaoKfold(x, y, test_size)
 

