import pandas as pd #type:ignore
import plotly.express as px #type:ignore
import plotly.graph_objects as go#type:ignore
import matplotlib.pyplot as plt#type:ignore
import seaborn as sns#type:ignore
import numpy as np#type:ignore

#px.set_mapbox_access_token(open("mapbox_token").read())
df_data = pd.read_csv("sao-paulo-properties-april-2019.csv")
print(df_data.head())