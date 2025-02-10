from sklearn.model_selection import cross_val_score,KFold, RandomizedSearchCV
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

  # model_linear = LinearRegression()
  model_elastic = ElasticNet()

  valores = {'alpha':[0.1, 0.5, 1, 2, 5, 10, 15, 20, 50, 100, 150, 200, 300, 500, 7550, 1000, 1500, 2000, 3000, 5000], 'l1_ratio': [0.02, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

  # kfold = KFold(n_splits=5)
  # resultado = cross_val_score(model_linear, x, y, cv=kfold)
  # print(resultado.mean())

  procura = RandomizedSearchCV(estimator=model_elastic, param_distributions=  valores,  n_iter= 150, cv= 5, random_state= 15)
  procura.fit(x, y)

  print(f'melhor score {procura.best_score_}')
  print(f'melhor alpha {procura.best_estimator_.alpha}')
  print(f'melhor li_ratio {procura.best_estimator_.l1_ratio}')