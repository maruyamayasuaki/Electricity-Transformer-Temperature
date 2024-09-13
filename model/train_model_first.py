import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# データの読み込みと前処理
df = pd.read_csv('../data/ett.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# 特徴量エンジニアリング
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
df['weekday'] = df.index.dayofweek

df = df.dropna() 

# スケーリング
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# 目的変数のデータの分割
X = df_scaled.drop('OT', axis=1).values
y = df_scaled['OT'].values

# LSTMモデルの定義
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# モデルの評価関数
def calculate_metrics(y_true, y_pred):
    return {
        'R-squared': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

# LSTMモデルのトレーニング関数
def train_lstm_model(model, X_train, y_train, X_test, y_test, n_epochs=300, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_train_pred = model(X_train)
        loss = criterion(y_train_pred, y_train)
        loss.backward()
        optimizer.step()
        
        # 評価モード
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test)
        
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {loss.item():.5f}, Test Loss: {test_loss.item():.5f}")

    # 損失の可視化
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.savefig('../data/loss.png')

    return train_losses, test_losses

# 時系列交差検証
tscv = TimeSeriesSplit(n_splits=5)
results = {}

input_dim = X.shape[1] # 特徴量の数

# モデルごとに評価
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_train_pred_lr = lr_model.predict(X_train)
    y_test_pred_lr = lr_model.predict(X_test)
    results[f'Linear Regression'] = {
        'train': calculate_metrics(y_train, y_train_pred_lr),
        'test': calculate_metrics(y_test, y_test_pred_lr)
    }

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)
    results[f'Random Forest'] = {
        'train': calculate_metrics(y_train, y_train_pred_rf),
        'test': calculate_metrics(y_test, y_test_pred_rf)
    }

    # LightGBM
    lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
    lgbm_model.fit(X_train, y_train)
    y_train_pred_lgbm = lgbm_model.predict(X_train)
    y_test_pred_lgbm = lgbm_model.predict(X_test)
    results[f'LightGBM'] = {
        'train': calculate_metrics(y_train, y_train_pred_lgbm),
        'test': calculate_metrics(y_test, y_test_pred_lgbm)
    }

    # ARIMA
    arima_train_scores = []
    arima_test_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = ARIMA(y_train, order=(1,1,1))
        fitted_model = model.fit()
        y_train_pred = fitted_model.fittedvalues
        y_test_pred = fitted_model.forecast(steps=len(y_test))
        
        arima_train_scores.append(calculate_metrics(y_train[1:], y_train_pred[1:]))  # ARIMAは1時点目から予測しているため、y_trainから1時点目以降を取得
        arima_test_scores.append(calculate_metrics(y_test, y_test_pred))

    results['ARIMA'] = {
        'train': pd.DataFrame(arima_train_scores).mean(),
        'test': pd.DataFrame(arima_test_scores).mean()
    }


    # LSTM
    X_train_lstm = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # (バッチサイズ, シーケンスの長さ, 特徴量の数)
    X_test_lstm = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train_lstm = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_lstm = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    lstm_model = LSTMModel(input_dim=input_dim, hidden_dim=128, num_layers=3, output_dim=1)
    train_losses, test_losses = train_lstm_model(lstm_model, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
    
    lstm_model.eval()
    with torch.no_grad():
        y_train_pred_lstm = lstm_model(X_train_lstm).numpy().flatten()
        y_test_pred_lstm = lstm_model(X_test_lstm).numpy().flatten()
    
    results[f'LSTM'] = {
        'train': calculate_metrics(y_train, y_train_pred_lstm),
        'test': calculate_metrics(y_test, y_test_pred_lstm)
    }

# 結果の表示
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"Train Metrics: {metrics['train']}")
    print(f"Test Metrics: {metrics['test']}")
