from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# Gerar dados
x, y = make_regression(n_samples=200, n_features=1, noise=30, random_state=42)

# Dividir os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=42)

# Testar diferentes configurações do Lasso
parametros = [
    {"alpha": 0.1, "max_iter": 1000, "tol": 0.01, "selection": "cyclic"},
    {"alpha": 1, "max_iter": 2000, "tol": 0.001, "selection": "cyclic"},
    {"alpha": 10, "max_iter": 5000, "tol": 0.0001, "selection": "cyclic"},
    {"alpha": 0.01, "max_iter": 1000, "tol": 0.1, "selection": "random"},
    {"alpha": 0.1, "max_iter": 2000, "tol": 0.01, "selection": "random"},
    {"alpha": 1, "max_iter": 3000, "tol": 0.001, "selection": "random"},
]

# Avaliar diferentes modelos
for i, params in enumerate(parametros, start=1):
    modelo = Lasso(**params)
    modelo.fit(x_treino, y_treino)
    score = modelo.score(x_teste, y_teste)
    print(f"Configuração {i}: {params}")
    print(f"Resultado do modelo Lasso: {score:.4f}\n")
