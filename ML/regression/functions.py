from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

  
from sklearn.model_selection import KFold

def modelosRegressao(x, y, test_size):
  models = {}
  m = []
  x_treino , x_teste, y_treino, y_teste = train_test_split(x, y, test_size= test_size)
  modeloLinear  =  LinearRegression()
  modeloLinear.fit(x_treino, y_treino)
  resultadoLinear = modeloLinear.score(x_teste, y_teste)
  
  models['regressao linear'] = resultadoLinear
  m.append(resultadoLinear)

  modeloRidge = Ridge(alpha=10)
  modeloRidge.fit(x_treino, y_treino)
  resultadoRidge = modeloRidge.score(x_teste, y_teste)
  models['regressao ridge'] = resultadoRidge
  m.append(resultadoRidge)

  modeloLasso = Lasso(alpha=10, max_iter=1000, tol= 0.01)
  modeloLasso.fit(x_treino, y_treino)
  resultadoLasso = modeloLasso.score(x_teste, y_teste)
  models['regressao lasso'] = resultadoLasso
  m.append(resultadoLasso)

  modeloElasticNet = ElasticNet(alpha=10, l1_ratio=0.5, tol= 2, max_iter=10000)
  modeloElasticNet.fit(x_treino, y_treino)
  resultadoElasticnet = modeloElasticNet.score(x_teste, y_teste)
  models['regressao elastic'] = resultadoElasticnet
  m.append(resultadoElasticnet)
  
  # max = 0 
  # model = ''
  for chave, valor in models.items():

    print(f"{chave} - {valor}")
  #   if valor > max:
  #     max = valor
  #     model = chave



  max_lst = max(models, key=models.get)

  print(f"O modelo com valor maximo é {max_lst} - {models[max_lst]}") 


def modelosRegressaoKfold(x, y, test_size):
  models = {}
  x_treino , x_teste, y_treino, y_teste = train_test_split(x, y, test_size= test_size)
  kfold = KFold(n_splits=10)

  modeloLinear  =  LinearRegression()
  modeloLinear.fit(x_treino, y_treino)
  resultadoLinear = cross_val_score(modeloLinear, x, y, cv=kfold)
  models['Regressao Linear'] = resultadoLinear.mean()


  modeloRidge = Ridge(alpha=10)
  modeloRidge.fit(x_treino, y_treino)
  resultadoRidge = cross_val_score(modeloRidge, x, y, cv=kfold)
  models['Regressao Ridge'] = resultadoRidge.mean()


  modeloLasso = Lasso(alpha=10, max_iter=1000, tol= 0.01)
  modeloLasso.fit(x_treino, y_treino)
  resultadoLasso = cross_val_score(modeloLasso, x, y, cv=kfold)
  models['Regressao Lasso'] = resultadoLasso.mean()


  modeloElasticNet = ElasticNet(alpha=10, l1_ratio=0.5, tol= 2, max_iter=10000)
  modeloElasticNet.fit(x_treino, y_treino)
  resultadoElasticnet = cross_val_score(modeloElasticNet, x, y, cv=kfold)
  
  models['Regressao Elastic'] = resultadoElasticnet.mean()
  print("\n --------------------------------------------------\n")
  for chave, valor in models.items():
    print(f"{chave} - {valor}")
  print("\n --------------------------------------------------\n")

  max_lst = max(models, key=models.get)
  print("\n --------------------------------------------------\n")
  print(f"O modelo com valor maximo é {max_lst} - {models[max_lst]}")   
  print("\n --------------------------------------------------\n")