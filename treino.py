import pandas as pd

# Dados dos treinos
treinos= {{"A": ["flexão 8 X 15", "flexão com elástico 8 x 15", "supino elástico 8 x 15",
                  "peito porção inferior 8 x 15", "puxada sentado 8 x 15", "puxada em pé 8 x 15",
                  "desenvolvimento 8 x 15", "deltoide 8 x 15", "trapézio 8 x 15"]}, {"B": ["abdominal crunch 8 x 10", "abdominal elevação 8 x 10",
                  "agachamento afundo 8 x 10", "agachamento lento 8 x 10",
                  "agachamento búlgaro quadríceps 8 x 10"]}}



# Criar DataFrames
df_treino_a = pd.DataFrame(treinos)


# Exportar para Excel
with pd.ExcelWriter("treinos.xlsx") as writer:
    df_treino_a.to_excel(writer, sheet_name="Treinos", index=False)
 

print("Planilha criada com sucesso!")

