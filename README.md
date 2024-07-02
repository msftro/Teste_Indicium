# Teste Cientista de Dados Indicium

## Instruções:

Antes de iniciar o projeto, certifique-se de rodar o seguinte comando em posse do arquivo **requiriments.txt** disponibilizado no repositório.

```plaintext
pip install -r requirements.txt
```

## 1. Faça uma análise exploratória dos dados (EDA), demonstrando as principais características entre as variáveis e apresentando algumas hipóteses relacionadas. Seja criativo!

- Primeiramente, comecei a análise pela coluna *Released_Year*, constatei que o filme Apollo 13	estava com o valor de ano e *Certificate* incorretos. Então fiz uma busca rápida na internet e prontamente corrigi os valores.
- A coluna *Certificate* possuia muitas variações de certificado, então resolvi unificar os certificados através dessa classficação:
  
1. Todas as Idades (U, G, Passed, Approved, TV-G)
2. Orientação dos Pais Recomendada (PG, TV-PG, GP, A)
3. Adolescentes e Acima (PG-13, UA, U/A, TV-14, 12)
4. Apenas Adultos (R, TV-MA, 16, Unrated)
5. Sem Classificação Definida (nan)

- Na coluna *Runtime* apenas removi o "min" e transformei em dado numérico.
- Na coluna *Gross* removi as vírgulas dos números e transformei novamente em dado numérico.
- Como haviam muitos dados faltantes em *Gross*, resolvi que a melhor solução seria substituir os valores ausentes pela mediana *Gross* de cada década. Dessa forma, os valores novos estariam mais condizentes com os valores de cada época. Como todos os valores de 2020 estavam também ausentes, decidi considerar como valores da década passada.
- Para a coluna *Meta Score* apenas substituí os valores ausentes pela mediana da coluna inteira.

  #### Colunas Genre, Overview, Director e StarN

  Devido a grande variedade de nomes disponíveis nessas colunas, decidi utilizar somente *Genre* para o treinamento e avaliação de modelo.

## 2. Responda também às seguintes perguntas:

#### a) Qual filme você recomendaria para uma pessoa que você não conhece?

Considerando que a pessoa se enquadre na faixa de classificação do filme. Consideraria indicar o filme que contém crime e ação no campo de gênero. Após isso orderania pelo campo de faturamento, pois se um filme faturou bastante significa que ele foi muito assistido (o que não significa necessariamente que é bom). Poderíamos também acrescentar as notas de crítica, mas sinceramente, a crítica está sempre errada. Portanto, com esses filtros aplicados, recomendaria: The Dark Knight	Action (Action, Crime, Drama).

#### b) Quais são os principais fatores que estão relacionados com alta expectativa de faturamento de um filme?

No modelo construído, os indicadores que mais influenciam no faturamento de um filme são os gêneros, sendo os três principais: Crime, Family e Action.

#### c) Quais insights podem ser tirados com a coluna Overview? É possível inferir o gênero do filme a partir dessa coluna?

Poderíamos extrair as palavras mais frequentes em filmes de maior faturamento para indicarmos ao departamento de *marketing*. Sim, mas acredito que necessite de uma metodologia mais avançada e que eu ainda não domino.

## 3. Explique como você faria a previsão da nota do imdb a partir dos dados. Quais variáveis e/ou suas transformações você utilizou e por quê? Qual tipo de problema estamos resolvendo (regressão, classificação)? Qual modelo melhor se aproxima dos dados e quais seus prós e contras? Qual medida de performance do modelo foi escolhida e por quê?

- Por meio de um modelo de *machine learning* que no caso é um *random forest*
- As variáveis utilizadas foram ['Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Meta_score', 'No_of_Votes']. Quando uma pessoa procura um filme para assistir, me incluindo nessa consideração, as principais características que eu busco são o gênero e o ano de lançamento. Geralmente eu não ligo muito para os outros parâmetros, mas eu sei que são características muito importantes para outras pessoas e que podem ajudar a classificar corretamente um modelo de previsão.
- Trata-se de um problema de regressão, pois estamos tentando prever o faturamento de um filme.
- O modelo escolhido foi um *random forest* que é excelente para modelos de regressão. É um modelo fácil de usar e relativamente preciso, porém necessita de uma base de dados grande, talvez por esse motivo ele não tenha performado muito bem. Entretanto, ele foi o escolhido pois ao longo dos testes que realizei os outros modelos desempenharam pior ou apresentaram *overfitting*. Utilizei como métricas: 'MSE','MAE','RMSE','R^2','MSLE' e 'MAPE'.

## 4. Supondo um filme com as seguintes características:

{'Series_Title': 'The Shawshank Redemption',
 'Released_Year': '1994',
 'Certificate': 'A',
 'Runtime': '142 min',
 'Genre': 'Drama',
 'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
 'Meta_score': 80.0,
 'Director': 'Frank Darabont',
 'Star1': 'Tim Robbins',
 'Star2': 'Morgan Freeman',
 'Star3': 'Bob Gunton',
 'Star4': 'William Sadler',
 'No_of_Votes': 2343110,
 'Gross': '28,341,469'}

#### Qual seria a nota do IMDB?

Ao rodar o arquivo test.py o modelo calculou a seguinte nota: 8.63991283

*Obs: As etapas de EDA deveriam ter sido feitas por pipeline para que os dados não necessitem ser tratados manualmente como foi feito na previsão.*

