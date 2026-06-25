import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pypfopt import EfficientFrontier, risk_models, expected_returns

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automated Investment Portfolio Management using ML")

# ─────────────────────────────────────────────
#  USER INPUT
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Enter Age", 18, 100, 22)
with col2:
    income = st.number_input("Enter Annual Income (₹)", 10000, 10000000, 500000)
with col3:
    risk = st.selectbox("Risk Level", ["Low", "Medium", "High"])

# ─────────────────────────────────────────────
#  STOCK DATA DOWNLOAD  ← FIX 1
#  Old code used data['Close'] on a multi-ticker
#  download which returns a MultiIndex DataFrame.
#  We now use group_by='ticker' and extract each
#  stock's Close column individually.
# ─────────────────────────────────────────────
stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
START  = '2021-01-01'
END    = '2026-01-01'

@st.cache_data(show_spinner="Downloading stock data…")
def load_prices(tickers, start, end):
    """Download multi-ticker data and return a clean Close-price DataFrame."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by='ticker',   # ← key fix: puts ticker as top-level key
        auto_adjust=True,
        progress=False,
        threads=True
    )
    # raw.columns is a MultiIndex: (ticker, OHLCV)
    # Extract Close for each ticker safely
    frames = {}
    for ticker in tickers:
        try:
            col = raw[ticker]['Close']
            frames[ticker] = col
        except KeyError:
            st.warning(f"Could not load data for {ticker}. Skipping.")
    df = pd.DataFrame(frames)
    df = df.ffill().dropna()
    return df

prices = load_prices(stocks, START, END)

if prices.empty:
    st.error("No stock data could be loaded. Please check your internet connection.")
    st.stop()

# ─────────────────────────────────────────────
#  STOCK PRICE TRENDS
# ─────────────────────────────────────────────
st.subheader("📈 Stock Price Trends")
st.line_chart(prices)

# ─────────────────────────────────────────────
#  RETURNS & CORRELATION HEATMAP  ← FIX 2
#  Previously failed because prices was malformed.
#  Now prices is a clean DataFrame so corr() works.
# ─────────────────────────────────────────────
returns = prices.pct_change().dropna()

st.subheader("🔥 Correlation Heatmap")
fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.heatmap(
    returns.corr(),
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    ax=ax1
)
ax1.set_title("Stock Return Correlation Matrix")
st.pyplot(fig1)
plt.close(fig1)

# ─────────────────────────────────────────────
#  ML MODEL COMPARISON (AAPL)
# ─────────────────────────────────────────────
st.subheader("🤖 ML Model Comparison (AAPL)")

@st.cache_data(show_spinner="Training models…")
def train_and_compare(price_series: pd.Series):
    df = price_series.to_frame(name='Close').copy()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Close']]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "Linear Regression"  : LinearRegression(),
        "Decision Tree"      : DecisionTreeRegressor(random_state=42),
        "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results   = []
    best_r2   = -np.inf
    best_model = None
    best_name  = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2   = r2_score(y_test, pred)
        results.append([name, round(mae, 2), round(rmse, 2), round(r2, 4)])
        if r2 > best_r2:
            best_r2    = r2
            best_model = model
            best_name  = name

    results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R² Score"])
    preds      = best_model.predict(X_test)
    return results_df, best_name, y_test.values, preds

results_df, best_model_name, y_actual, y_pred = train_and_compare(prices['AAPL'])

st.write("### Model Performance")
st.dataframe(results_df, use_container_width=True)
st.success(f"✅ Best Model: **{best_model_name}**")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(y_actual, label="Actual Price",    color='steelblue')
ax2.plot(y_pred,   label="Predicted Price", color='orangered', linestyle='--')
ax2.set_title(f"AAPL Price Prediction — {best_model_name}")
ax2.set_xlabel("Test Days")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)
plt.close(fig2)

# ─────────────────────────────────────────────
#  ML STOCK RECOMMENDATION  ← FIX 3
#  Old loop used yf.download() with multiple
#  tickers inside the loop, hitting the same
#  MultiIndex issue. Now uses single-ticker
#  download which returns simple columns.
# ─────────────────────────────────────────────
st.subheader("💡 ML Based Stock Recommendation")

@st.cache_data(show_spinner="Generating recommendations…")
def get_recommendations(tickers, price_df: pd.DataFrame):
    recs = []
    for stock in tickers:
        try:
            col = price_df[stock].dropna()
            if len(col) < 50:
                continue

            df = col.to_frame(name='Close').copy()
            df['Target'] = df['Close'].shift(-1)
            df.dropna(inplace=True)

            X = df[['Close']]
            y = df['Target']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            current_price   = float(df['Close'].iloc[-1])
            future_input    = pd.DataFrame([[current_price]], columns=['Close'])
            predicted_price = float(model.predict(future_input)[0])
            expected_return = ((predicted_price - current_price) / current_price) * 100

            recs.append([stock, round(current_price, 2), round(predicted_price, 2), round(expected_return, 2)])

        except Exception as e:
            st.warning(f"{stock}: {e}")

    if not recs:
        return pd.DataFrame()

    rec_df = pd.DataFrame(recs, columns=["Stock", "Current Price ($)", "Predicted Price ($)", "Predicted Return (%)"])
    rec_df = rec_df.sort_values("Predicted Return (%)", ascending=False).reset_index(drop=True)
    return rec_df

rec_df = get_recommendations(stocks, prices)

if not rec_df.empty:
    st.dataframe(rec_df, use_container_width=True)
    top_stock = rec_df.iloc[0]['Stock']
    top_ret   = rec_df.iloc[0]['Predicted Return (%)']
    st.success(f"🏆 Top Recommended Stock: **{top_stock}** (Expected Return: {top_ret}%)")

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
    colors = ['green' if r >= 0 else 'red' for r in rec_df["Predicted Return (%)"]]
    ax_bar.bar(rec_df["Stock"], rec_df["Predicted Return (%)"], color=colors)
    ax_bar.axhline(0, color='black', linewidth=0.8)
    ax_bar.set_title("ML Predicted Return by Stock")
    ax_bar.set_ylabel("Predicted Return (%)")
    st.pyplot(fig_bar)
    plt.close(fig_bar)
else:
    st.warning("Could not generate recommendations.")

# ─────────────────────────────────────────────
#  RISK PROFILE & PORTFOLIO PIE CHART
# ─────────────────────────────────────────────
st.subheader("📂 Recommended Portfolio Allocation")

if risk == "Low":
    portfolio = {"Bonds": 60, "ETF": 30, "Stocks": 10}
elif risk == "Medium":
    portfolio = {"Stocks": 50, "ETF": 30, "Bonds": 20}
else:
    portfolio = {"Stocks": 70, "ETF": 20, "Crypto": 10}

col_t, col_p = st.columns([1, 1])
with col_t:
    st.write(f"**Risk Level:** {risk}")
    for asset, pct in portfolio.items():
        st.write(f"- **{asset}:** {pct}%")

with col_p:
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    wedge_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    ax3.pie(
        portfolio.values(),
        labels=portfolio.keys(),
        autopct="%1.1f%%",
        startangle=140,
        colors=wedge_colors[:len(portfolio)]
    )
    ax3.set_title(f"Portfolio Allocation — {risk} Risk")
    st.pyplot(fig3)
    plt.close(fig3)

# ─────────────────────────────────────────────
#  MARKOWITZ PORTFOLIO OPTIMIZATION  ← FIX 4
#  Old code ran dropna on mu and S separately,
#  which breaks their index alignment and causes
#  EfficientFrontier to fail. We now build a
#  clean prices_opt from the already-fixed prices
#  DataFrame and pass mu/S directly without
#  reshaping them.
# ─────────────────────────────────────────────
st.subheader("📐 Markowitz Portfolio Optimization")

try:
    # Use the already-clean prices DataFrame
    prices_opt = prices.copy()
    prices_opt = prices_opt.replace([np.inf, -np.inf], np.nan)
    prices_opt = prices_opt.ffill().dropna()

    if prices_opt.shape[0] < 30:
        st.warning("Not enough data for Markowitz optimization (need 30+ trading days).")
    else:
        mu = expected_returns.mean_historical_return(prices_opt)
        S  = risk_models.sample_cov(prices_opt)

        # Do NOT dropna on mu/S separately — it breaks index alignment
        # Just feed them directly into EfficientFrontier
        ef = EfficientFrontier(mu, S)
        raw_weights     = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        # Display weights
        weight_df = pd.DataFrame(
            list(cleaned_weights.items()),
            columns=["Stock", "Weight"]
        )
        weight_df["Weight (%)"] = (weight_df["Weight"] * 100).round(2)
        weight_df = weight_df[weight_df["Weight (%)"] > 0]

        col_w, col_wp = st.columns([1, 1])
        with col_w:
            st.write("**Optimized Weights**")
            st.dataframe(weight_df[["Stock", "Weight (%)"]], use_container_width=True)

            exp_ret, volatility, sharpe = ef.portfolio_performance()
            st.metric("Expected Annual Return", f"{exp_ret:.2%}")
            st.metric("Annual Volatility",      f"{volatility:.2%}")
            st.metric("Sharpe Ratio",           f"{sharpe:.2f}")

        with col_wp:
            fig4, ax4 = plt.subplots(figsize=(5, 5))
            ax4.pie(
                weight_df["Weight (%)"],
                labels=weight_df["Stock"],
                autopct="%1.1f%%",
                startangle=140
            )
            ax4.set_title("Markowitz Optimal Weights")
            st.pyplot(fig4)
            plt.close(fig4)

except Exception as e:
    st.error(f"Optimization Error: {e}")

# ─────────────────────────────────────────────
#  PROJECT SUMMARY
# ─────────────────────────────────────────────
st.subheader("✅ Project Summary")
summary_items = [
    "Stock Data Download (yfinance, fixed MultiIndex handling)",
    "Stock Price Trend Visualization",
    "Correlation Heatmap",
    "ML Model Comparison — Linear, Decision Tree, Random Forest",
    "AAPL Price Prediction with best model",
    "ML Stock Recommendation with predicted return bar chart",
    "Risk-based Portfolio Allocation with pie chart",
    "Markowitz Portfolio Optimization (Sharpe ratio maximized)",
]
for item in summary_items:
    st.write(f"✔ {item}")
