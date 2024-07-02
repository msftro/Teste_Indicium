# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv('data/desafio_indicium_imdb.csv')
df.drop(columns='Unnamed: 0', inplace=True)
df.head()

# %%
df.info()

# %%
df['Released_Year'].unique()

# %%
df[df['Released_Year'] == 'PG']

# %%
#Por meio de pesquisa na internet identifiquei o ano correto e a classificação

df.iloc[965, 1] = 1995
df.iloc[965, 2] = 'PG'

# %%
df[df['Series_Title'] == 'Apollo 13']

# %%
df['Released_Year'].unique()

# %%
df['Certificate'].unique()

# %%
#Unificar as categorias

certificate_map = {
    'U': 1,
    'G': 1,
    'Passed': 1,
    'Approved': 1,
    'TV-G': 1,
    'PG': 2,
    'TV-PG': 2,
    'GP': 2,
    'A': 2,  # Supondo que A signifique "Todos" em alguns contextos
    'PG-13': 3,
    'UA': 3,
    'U/A': 3,
    'TV-14': 3,
    'R': 4,
    'TV-MA': 4,
    '16': 4,
    'Unrated': 4,
    np.nan: 5
}

# %%
df['Certificate'] = df['Certificate'].map(certificate_map)
df.head()

# %%
df['Runtime'].unique()

# %%
df['Runtime'] = df['Runtime'].str.replace(' min', '')
df.head()

# %%
df['Gross'] = df['Gross'].str.replace(',', '')
df.head()

# %%
df['Released_Year'] = pd.to_datetime(df['Released_Year'], format='%Y')
df['Released_Year'] = df['Released_Year'].dt.year

# %%
df.info()

#%%
df_year_clean_1 = df.copy()
df_year_clean_1.replace(2020, 2019, inplace=True)
df_year_clean_1['new_released_year'] = df_year_clean_1['Released_Year'].astype(str)\
                                        .apply(lambda x: x[:-1] + '0').astype(int)

# %%
df_year_clean_2 = df_year_clean_1.copy()
df_year_clean_2.dropna(subset=['Gross'], inplace=True)
df_year_clean_2['Gross'] = df_year_clean_2['Gross'].astype(int)

release_decade_median = df_year_clean_2.groupby('new_released_year')\
                                        .agg({'Gross':'median'})
release_decade_median

#%%
df_year_clean_1 = df_year_clean_1.merge(release_decade_median, 
                                       left_on='new_released_year', 
                                       right_index=True, 
                                       how='left')
df_year_clean_1[:50]

#%%
df_year_clean_1.fillna({'Gross_x': df_year_clean_1['Gross_y']}, inplace=True)
df_year_clean_1[:50]

# %%
df['Gross'] = df_year_clean_1['Gross_x']
df[:50]

# %%
df.info()

# %%
meta_median = df['Meta_score'].median()
df.fillna({'Meta_score':meta_median}, inplace=True)

# %%
df.info()

# %%
df['Released_Year'] = df['Released_Year'].astype(int)
df['Runtime'] = df['Runtime'].astype(int)
df['Gross'] = df['Gross'].astype(int)

df.info()

# %%
df[:50]

# %%
df.to_csv('data/cleaned_data.csv', index=False)

# %%