import pandas as pd
import gdown
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Importando os dados
file_id = '1t_3xs96ce44UiVYJ8KqG3hOxRWllXZYy'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'data.csv'
gdown.download(url, output, quiet=False)
df = pd.read_csv('data.csv', sep=',')

# Excluindo colunas desnecessárias
df = df.drop('Unnamed: 0', axis=1)

# Transformando a coluna runtime em apenas números inteiros
df['Runtime'] = df['Runtime'].str.replace('min', '').astype(int)

# Separação das varíaveis categóricas e numéricas
categorical_cols = [
    'Series_Title', 'Released_Year', 'Certificate', 'Genre', 'Overview',
    'Director', 'Star1', 'Star2', 'Star3', 'Star4'
]

numerical_cols = [
    'IMDB_Rating', 'Runtime', 'Meta_score', 'No_of_Votes', 'Gross'
]

# Verificando e tratando quantidade de nulos
# df.isnull().sum()

#Transformar ano de lançamento em númerico
#   Apresenta erro, pois o filme apollo 13 está com seu ano de lançamento como
#   'PG', após pesquisa foi verificado que seu ano de lançamento é 1995, vamos
#   substituir isso no dataset:
df.loc[df['Series_Title'] == 'Apollo 13', 'Released_Year'] = 1995
df[df['Series_Title'] == 'Apollo 13']
df['Released_Year'] = df['Released_Year'].astype('int64')

df = df.dropna(subset=['Gross', 'Meta_score', 'Certificate'])
# df.isnull().sum()

# Convertendo colunas numéricas com vírgulas para tipo Float
for col in numerical_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '', regex=True).astype(float)

# Selecionando as colunas relevantes
x = df[['Released_Year', 'Certificate', 'Runtime', 'Genre', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross']]
y = df['IMDB_Rating']

# Codificar variáveis categóricas
categorical_cols = ['Certificate', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']
numerical_cols = ['Released_Year', 'Runtime', 'Meta_score', 'No_of_Votes', 'Gross']

# Pré-processamento utilizando StandardScaler e OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Dividindo os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=30),)
    ])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print(f'Random Forest Regression - MSE: {mse}, RMSE: {rmse}, MAE: {mae}')

with open("model.pkl", "wb") as f:
     pickle.dump(pipeline, f)