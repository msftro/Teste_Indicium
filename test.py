#%%
import pandas as pd
import numpy as np
import pickle

#%%
# Carregar o modelo exportado
with open('data/LH_CD_MARCIOFERREIRA.pkl', 'rb') as f:
    model_export = pickle.load(f)

#%%
# Extrair o modelo e os recursos do modelo exportado
model = model_export['model']
features = model_export['features']

#%%
# Criar o dataframe com os dados a serem previstos
data_to_predict = {
    'Released_Year': [1994],
    'Certificate': [2],
    'Runtime': [142],
    'Genre': ['Drama'],
    'Meta_score': [80.0],
    'No_of_Votes': [2343110],
    'Gross': [28341469]
}
X_to_predict = pd.DataFrame(data_to_predict)

#%%
# Realizar a previsão
predicted_rating = model.predict(X_to_predict)

print("Predicted IMDB Rating:", predicted_rating)

# %%
