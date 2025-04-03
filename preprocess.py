import pandas as pd
import numpy as np
import statsmodels.api as sm
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numba import jit, prange


@jit(nopython=True)
def calculate_rolling_std(arr, output, window):
  n = len(arr)
  for i in range(n - window + 1):
      output[i + window - 1] = np.std(arr[i:i + window])
  return output

@jit(nopython=True)
def calc_lstsq(X, y):
  # Solve X'X b = X'y
  XTX = X.T @ X
  XTy = X.T @ y

  # Check if XTX is invertible
  det = XTX[0, 0] * XTX[1, 1] - XTX[0, 1] * XTX[1, 0]
  if abs(det) < 1e-10:
      return np.array([np.nan, np.nan])

  # Inverse of 2x2 matrix
  inv_XTX = np.empty((2, 2))
  inv_XTX[0, 0] = XTX[1, 1] / det
  inv_XTX[1, 1] = XTX[0, 0] / det
  inv_XTX[0, 1] = -XTX[0, 1] / det
  inv_XTX[1, 0] = -XTX[1, 0] / det

  # Calculate beta
  beta = inv_XTX @ XTy
  return beta

@jit(nopython=True)
def rolling_cointegration_score(asset1, asset2, window=60, eps=1e-10):
  n = len(asset1)
  if n < window:
    raise ValueError("Window size too large for available data")

  scores = np.full(n, np.nan)

  for i in range(n - window + 1):
    y = asset1[i:i + window]
    x_vals = asset2[i:i + window]
    y_std = np.std(y)
    x_std = np.std(x_vals)

    if y_std < eps or x_std < eps:
      continue

    X = np.column_stack((np.ones(window), x_vals))
    beta = calc_lstsq(X, y)

    if np.isnan(beta[0]):
      continue

    residuals = y - (beta[0] + beta[1] * x_vals)

    if np.std(residuals) < eps:
      continue

    adf_stat = -np.std(residuals) / np.mean(np.abs(residuals))
    scores[i + window - 1] = adf_stat

  return scores

@jit(nopython=True)
def calculate_rolling_hedge_ratio_numba(asset1_prices, asset2_prices, window_size=30):
  n = len(asset1_prices)
  hedge_ratios = np.zeros(n)
  hedge_ratios[:window_size-1] = np.nan

  for i in range(window_size-1, n):
    y = asset1_prices[i-window_size+1:i+1]
    x = asset2_prices[i-window_size+1:i+1]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = 0.0
    denominator = 0.0

    for j in range(window_size):
      x_diff = x[j] - x_mean
      y_diff = y[j] - y_mean
      numerator += x_diff * y_diff
      denominator += x_diff * x_diff

    if denominator != 0:
      hedge_ratios[i] = numerator / denominator
    else:
      hedge_ratios[i] = np.nan

  return hedge_ratios

if __name__ == "__main__":
  eudf = pd.read_csv("../Data/EURUSD.csv", index_col="Gmt time")
  gbdf = pd.read_csv("../Data/GBPUSD.csv", index_col="Gmt time")
  eudf.index = pd.to_datetime(eudf.index, format="%d.%m.%Y %H:%M:%S.%f")
  gbdf.index = pd.to_datetime(gbdf.index, format="%d.%m.%Y %H:%M:%S.%f")
  common_index = eudf.index.intersection(gbdf.index)
  eudf = eudf.reindex(common_index)
  gbdf = gbdf.reindex(common_index)
  ptd = pd.DataFrame(index=common_index)
  ptd.index.name = "Gmt time"
  ptd["asset1_price"] = eudf["Close"]
  ptd["asset2_price"] = gbdf["Close"]

  ptd["open"] = eudf['Open'] / gbdf['Open']
  ptd["high"] = eudf['High'] / gbdf['High']
  ptd["low"] = eudf['Low'] / gbdf['Low']
  ptd["close"] = eudf['Close'] / gbdf['Close']
  ptd['ratio_price'] = eudf['Close'] / gbdf['Close']
  
  ptd["hedge_ratio"] = calculate_rolling_hedge_ratio_numba(np.array(ptd["asset1_price"]), np.array(ptd["asset2_price"]))
  spread = ptd["asset1_price"] - ptd["asset2_price"]
  ptd["spread_zscore"] = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
  
  
  ptd["rolling_correlation"] = ptd["asset1_price"].rolling(30).corr(ptd["asset2_price"])
  coint_scores = rolling_cointegration_score(ptd["asset1_price"].values, ptd["asset2_price"].values)
  ptd["rolling_cointegration_score"] = coint_scores
  
  ptd["RSI1"] = RSIIndicator(ptd["asset1_price"], window=14).rsi()
  ptd["RSI2"] = RSIIndicator(ptd["asset2_price"], window=14).rsi()
  ptd["RSI3"] = RSIIndicator(ptd["ratio_price"], window=14).rsi()
  
  ptd["MACD1"] = MACD(ptd["asset1_price"], window_slow=26, window_fast=12, window_sign=9).macd()
  ptd["MACD2"] = MACD(ptd["asset2_price"], window_slow=26, window_fast=12, window_sign=9).macd()
  ptd["MACD3"] = MACD(ptd["ratio_price"], window_slow=26, window_fast=12, window_sign=9).macd()
  
  scaler = MinMaxScaler()

  scale_columns = ['open', 'high', 'low', 'close',
         'ratio_price', 'spread_zscore', 'rolling_correlation',
         'rolling_cointegration_score', 'RSI1', 'RSI2', 'RSI3', 'MACD1', 'MACD2',
         'MACD3']
  
  ptds = pd.DataFrame(index=ptd.index)
  ptds[scale_columns] = ptd[scale_columns]
  ptds = ptds.replace([np.inf, -np.inf], np.nan).dropna()
  
  tds = pd.DataFrame(scaler.fit_transform(ptds), columns=scale_columns)
  
  ptd.dropna(inplace=True)
  ptd.to_csv("../data/processed/tData.csv")