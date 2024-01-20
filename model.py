from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
import pandas as pd
import config as conf

params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learnnig_rage': 0.05,
    'metric': {'l2','l1'}
}

def train(df):
    X = get_X(df, conf.past, conf.h)
    y = get_y(df, conf.past, conf.h)

    print(X.head())
    print(y.head())

    model = LGBMRegressor(random_state=42)
    model.fit(X, y)
    model.score(X, y)
    print(r2_score(y, model.predict(X)))

    return model

def predict(model, df):
    X = get_X(df, conf.past, conf.h) 
    y = model.predict(X)

    return y

def get_X(df, past, h):
    df = df.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    df = df[past:-h]
    return df.reset_index(drop=True)

def get_y(df, past, h):
    y = df['Close'].to_numpy()
    y = [y[i+h]/y[i]-1 for i in range(0, len(y)-h)]
    return pd.DataFrame(y[past:])

    
    
