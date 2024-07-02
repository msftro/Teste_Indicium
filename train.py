# %%
import pandas as pd
import numpy as np

from feature_engine.pipeline import Pipeline

from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

# %%
df = pd.read_csv('data/cleaned_data.csv')
df.head()

# %%
df.columns

# %%
features = ['Released_Year', 'Certificate', 'Runtime', 'Genre',
            'Meta_score', 'No_of_Votes', 'Gross']
target = 'IMDB_Rating'

X = df[features]
y = df[target]

# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                    y, 
                                                                    test_size=0.2, 
                                                                    random_state=42)

# %%
class GenreTokenizer(BaseEstimator, TransformerMixin): 
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None): 
        self.mlb.fit(X['Genre'].str.split(', '))
        return self 

    def transform(self, X, y=None):
        genres_encoded = self.mlb.transform(X['Genre'].str.split(', '))
        genres_df = pd.DataFrame(genres_encoded, columns=self.mlb.classes_, index=X.index)
        X = X.drop('Genre', axis=1)
        return pd.concat([X, genres_df], axis=1)

# %%
model = ensemble.RandomForestRegressor(random_state=42)

params = {
    'max_depth': [4, 5, 8, 10, 15],
    'min_samples_leaf': [10, 15, 20, 50, 100],
    'n_estimators': [100, 200, 500]
}

grid = model_selection.GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', n_jobs=-1)

pipe = Pipeline([
    ('genre_tokenizer', GenreTokenizer()),
    ('grid_search', grid)
])

pipe.fit(X_train, y_train)

# %%
train_pred = pipe.predict(X_train)
test_pred = pipe.predict(X_test)

# Mean Squared Error
mse_train = metrics.mean_squared_error(y_train, train_pred)
mse_test = metrics.mean_squared_error(y_test, test_pred)

# Mean Absolute Error
mae_train = metrics.mean_absolute_error(y_train, train_pred)
mae_test = metrics.mean_absolute_error(y_test, test_pred)

# Root Mean Squared Error
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# R² Score
r2_train = metrics.r2_score(y_train, train_pred)
r2_test = metrics.r2_score(y_test, test_pred)

# Mean Squared Logarithmic Error
msle_train = metrics.mean_squared_log_error(y_train, train_pred)
msle_test = metrics.mean_squared_log_error(y_test, test_pred)

# Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_train = mean_absolute_percentage_error(y_train, train_pred)
mape_test = mean_absolute_percentage_error(y_test, test_pred)

print('Train Metrics:')
print('MSE:', mse_train)
print('MAE:', mae_train)
print('RMSE:', rmse_train)
print('R^2:', r2_train)
print('MSLE:', msle_train)
print('MAPE:', mape_train)

print('\nTest Metrics:')
print('MSE:', mse_test)
print('MAE:', mae_test)
print('RMSE:', rmse_test)
print('R^2:', r2_test)
print('MSLE:', msle_test)
print('MAPE:', mape_test)

metrics_values = {
    'train': {
        'MSE': mse_train,
        'MAE': mae_train,
        'RMSE': rmse_train,
        'R^2': r2_train,
        'MSLE': msle_train,
        'MAPE': mape_train
    },
    'test': {
        'MSE': mse_test,
        'MAE': mae_test,
        'RMSE': rmse_test,
        'R^2': r2_test,
        'MSLE': msle_test,
        'MAPE': mape_test
    }
}

model_export = pd.Series({
    'model': pipe,
    'features': features,
    'metrics': metrics_values
})

# %%
model_export.to_pickle('data/LH_CD_MARCIOFERREIRA.pkl')

# %%
