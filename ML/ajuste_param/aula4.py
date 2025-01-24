from sklearn.model_selection import cross_val_score,  KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
import pandas as pd

if "__main__" == __name__:
  # x, y = make_regression(n_samples=200, n_features=1, noise=30)
  # modelosRegressao(x, y)
  colunas = ['No' , 'gre' , 'toefl', 'rating', 'sop', 'lor', 'cgpa', 'research', 'chance']
  pd.set_option("display.max_columns", 320)
  dados = pd.read_csv("archive4/Admission_Predict_Ver1.1.csv", names = colunas)
  # dados = pd.read_csv("kc_house_data.csv")
  dados = dados.drop(0)
  dados.drop('No', axis=1, inplace=True)

  y = dados["chance"]
  x = dados.drop('chance', axis=1)

  model_linear = LinearRegression()
  kfold = KFold(n_splits=5)
  resultado = cross_val_score(model_linear, x, y, cv=kfold)
  print(resultado.mean())

