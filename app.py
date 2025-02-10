import pandas as pd
from dash import Dash, dcc, html, Input, Output

# Carregar os dados
df = pd.read_csv("ML/avocado.csv")

# Filtrar os dados para Albany e abacates convencionais
data = (
    df.query("type == 'conventional' and region == 'Albany'")
    .assign(Date=lambda df: pd.to_datetime(df["Date"], format="%Y-%m-%d"))
    .sort_values(by="Date")
)

# Dados agregados para gráficos específicos
volume_por_ano = data.groupby("year", as_index=False)["Total Volume"].sum()
categorias = data[["4046", "4225", "4770"]].sum()

# Inicializar o aplicativo Dash
app = Dash(__name__)

# Layout do aplicativo com abas
app.layout = html.Div([
    html.H1("Dashboard de Vendas de Abacate", style={"textAlign": "center"}),

    # Criando abas
    dcc.Tabs(id="tabs", value="linha", children=[
        dcc.Tab(label="Preço Médio ao Longo do Tempo", value="linha"),
        dcc.Tab(label="Volume de Vendas por Ano", value="barras"),
        dcc.Tab(label="Distribuição das Categorias", value="pizza"),
    ]),

    # Área para exibição dos gráficos
    html.Div(id="conteudo-abas")
])

# Callback para alternar os gráficos com base na aba selecionada
@app.callback(
    Output("conteudo-abas", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab_escolhida):
    if tab_escolhida == "linha":
        return dcc.Graph(
            figure={
                "data": [
                    {"x": data["Date"], "y": data["AveragePrice"], "type": "line", "name": "Preço Médio"}
                ],
                "layout": {"title": "Evolução do Preço Médio do Abacate"}
            }
        )

    elif tab_escolhida == "barras":
        return dcc.Graph(
            figure={
                "data": [
                    {"x": volume_por_ano["year"], "y": volume_por_ano["Total Volume"], "type": "bar", "name": "Volume Total"}
                ],
                "layout": {"title": "Total de Abacates Vendidos por Ano"}
            }
        )

    elif tab_escolhida == "pizza":
        return dcc.Graph(
            figure={
                "data": [
                    {"values": categorias.values, "labels": categorias.index, "type": "pie", "name": "Categorias"}
                ],
                "layout": {"title": "Proporção de Vendas por Categoria"}
            }
        )

# Executar o aplicativo
if __name__ == "__main__":
    app.run_server(debug=True)
