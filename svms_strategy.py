
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def heston_simulation(S0, V0, r, rho, kappa, theta, sigma, nu, T, dt):
    N = int(T/dt)
    S = np.zeros(N)
    V = np.zeros(N)
    S[0] = S0
    V[0] = V0
    Z1 = np.random.randn(N)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(N)
    for t in range(1, N):
        S[t] = S[t-1] + r * S[t-1] * dt + np.sqrt(V[t-1]) * S[t-1] * np.sqrt(dt) * Z1[t]
        V[t] = V[t-1] + kappa * (theta - V[t-1]) * dt + sigma * np.sqrt(V[t-1]) * np.sqrt(dt) * Z2[t]
        V[t] = max(V[t], 0)
    return S, V

def compute_RSI(data, window=14):
    delta = data.diff().dropna()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    dates = pd.date_range(start="2022-01-01", periods=252, freq="D")
    S0 = 100
    V0 = 0.04
    r = 0.05
    rho = -0.2
    kappa = 2.0
    theta = 0.04
    sigma = 0.2
    nu = 0.1
    T = 1.0
    dt = 1/252
    heston_S, heston_V = heston_simulation(S0, V0, r, rho, kappa, theta, sigma, nu, T, dt)
    rsi = compute_RSI(pd.Series(heston_S))
    heston_V_adjusted = heston_V[1:]
    buy_signals = (rsi < 30) & (np.sqrt(heston_V_adjusted) > np.sqrt(V0))
    sell_signals = (rsi > 70) & (np.sqrt(heston_V_adjusted) < np.sqrt(V0))
    plt.figure(figsize=(14, 7))
    plt.plot(dates[1:], heston_S[1:], label="Stock Price", color="blue")
    plt.scatter(dates[1:][buy_signals], heston_S[1:][buy_signals], color="green", label="Buy Signal", marker="^")
    plt.scatter(dates[1:][sell_signals], heston_S[1:][sell_signals], color="red", label="Sell Signal", marker="v")
    plt.title("Stochastic Volatility Momentum Strategy")
    plt.legend()
    plt.grid(True)
    plt.show()

