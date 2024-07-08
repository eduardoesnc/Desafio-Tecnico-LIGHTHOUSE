import pandas as pd
import numpy as np
import pickle

dados = {
    'Series_Title': 'The Shawshank Redemption',
    'Released_Year': 1994,
    'Certificate': 'A',
    'Runtime': 142,
    'Genre': 'Drama',
    'Meta_score': 80.0,
    'Director': 'Frank Darabont',
    'Star1': 'Tim Robbins',
    'Star2': 'Morgan Freeman',
    'Star3': 'Bob Gunton',
    'Star4': 'William Sadler',
    'No_of_Votes': 2343110,
    'Gross': 28341469
}

dados_df = pd.DataFrame([dados])

#Carregar modelo
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)
    
predictions = pipeline.predict(dados_df)

print("Previsão da nota IMDB:")
print(f"{predictions[0]:.1f}")