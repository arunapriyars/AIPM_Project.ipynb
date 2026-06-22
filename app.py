import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pypfopt import EfficientFrontier, risk_models, expected_returns

st.title("📊 Automated Investment Portfolio Management using ML")

# ---------------- USER INPUT ----------------

age = st.number_input("Enter Age", 18, 100, 22)
income = st.number_input("Enter Annual Income", 10000, 10000000, 500000)
risk = st.selectbox("Risk Level", ["Low", "Medium", "High"])

# ---------------- STOCK DATA ----------------

stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']

data = yf.download(
    stocks,
    start='2021-01-01',
    end='2026-01-01',
    auto_adjust=True
)

prices = data['Close']

# Clean data
prices = prices.ffill().dropna()

# ---------------- STOCK PRICE GRAPH ----------------

st.subheader("Stock Price Trends")
st.line_chart(prices)

# ---------------- RETURNS ----------------

returns = prices.pct_change().dropna()

# ---------------- HEATMAP ----------------

st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    returns.corr(),
    annot=True,
    cmap='coolwarm',
    ax=ax
)

st.pyplot(fig)

# ---------------- ML SECTION ----------------

st.subheader("ML Stock Price Prediction (AAPL)")

aapl = yf.download(
    'AAPL',
    start='2021-01-01',
    end='2026-01-01',
    auto_adjust=True
)

aapl = aapl[['Close']].copy()

aapl['Target'] = aapl['Close'].shift(-1)

aapl.dropna(inplace=True)

X = aapl[['Close']]
y = aapl['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.plot(
    y_test.values,
    label="Actual Price"
)

ax2.plot(
    predictions,
    label="Predicted Price"
)

ax2.set_title("AAPL Stock Prediction")
ax2.legend()

st.pyplot(fig2)

# ---------------- RISK PROFILE ----------------

st.subheader("Recommended Portfolio")

if risk == "Low":
    portfolio = {
        "Bonds": 60,
        "ETF": 30,
        "Stocks": 10
    }

elif risk == "Medium":
    portfolio = {
        "Stocks": 50,
        "ETF": 30,
        "Bonds": 20
    }

else:
    portfolio = {
        "Stocks": 70,
        "ETF": 20,
        "Crypto": 10
    }

st.write(portfolio)

# ---------------- PIE CHART ----------------

fig3, ax3 = plt.subplots(figsize=(6, 6))

ax3.pie(
    portfolio.values(),
    labels=portfolio.keys(),
    autopct="%1.1f%%"
)

ax3.set_title("Portfolio Allocation")

st.pyplot(fig3)

# ---------------- MARKOWITZ OPTIMIZATION ----------------

st.subheader("Markowitz Portfolio Optimization")

try:

    prices_opt = prices.copy()

    prices_opt = prices_opt.replace(
        [np.inf, -np.inf],
        np.nan
    )

    prices_opt = prices_opt.ffill().dropna()

    mu = expected_returns.mean_historical_return(prices_opt)

    S = risk_models.sample_cov(prices_opt)

    mu = mu.dropna()

    S = S.dropna(how="all").dropna(axis=1, how="all")

    common_assets = mu.index.intersection(S.index)

    mu = mu.loc[common_assets]
    S = S.loc[common_assets, common_assets]

    ef = EfficientFrontier(mu, S)

    weights = ef.max_sharpe()

    cleaned_weights = ef.clean_weights()

    st.write("Optimized Weights")

    st.write(cleaned_weights)

    expected_return, volatility, sharpe = ef.portfolio_performance()

    st.write(f"Expected Annual Return: {expected_return:.2%}")
    st.write(f"Annual Volatility: {volatility:.2%}")
    st.write(f"Sharpe Ratio: {sharpe:.2f}")

except Exception as e:

    st.error(f"Optimization Error: {e}")

# ---------------- PROJECT SUMMARY ----------------

st.subheader("Project Summary")

st.write("✔ Stock Data Download")
st.write("✔ Data Visualization")
st.write("✔ Correlation Analysis")
st.write("✔ ML Stock Prediction")
st.write("✔ Risk Profiling")
st.write("✔ Portfolio Recommendation")
st.write("✔ Markowitz Portfolio Optimization")
