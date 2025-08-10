# GMF Investments – Week 11 Interim Report

## Task 1: Data Preprocessing, EDA, Stationarity & Risk Analysis

This report covers the first phase of the GMF Investments time series forecasting and portfolio optimization project. The focus of Task 1 is to fetch historical financial data for TSLA, BND, and SPY; preprocess it; conduct exploratory data analysis (EDA); check for stationarity; and compute basic risk metrics such as Value at Risk (VaR) and Sharpe Ratios.

## 1. Data Sources & Description

Data was sourced from Yahoo Finance (via the yfinance API) for the period **July 1, 2015, to July 31, 2025**:

- **TSLA**: High-growth, high-volatility stock (Automobile Manufacturing)
- **BND**: Vanguard Total Bond Market ETF, low-volatility fixed income exposure
- **SPY**: S&P 500 ETF, diversified U.S. equity market exposure

## 2. Data Preprocessing

Steps performed:

- Loaded Adjusted Close prices for all assets
- Checked and handled missing values
- Computed daily returns using percentage change
- Aligned data by trading days across all assets
- Saved cleaned datasets for later modeling

## 3. Exploratory Data Analysis

![Adjusted Close Prices](prices.png)  
_Figure 1: Adjusted Close Prices for TSLA, BND, and SPY over the analysis period._

![Daily Returns](returns.png)  
_Figure 2: Daily returns for TSLA, BND, and SPY. TSLA exhibits the highest volatility._

![Rolling 20-Day Volatility](rolling_vol.png)  
_Figure 3: 20-day rolling volatility. TSLA shows pronounced spikes compared to SPY and BND._

## 4. Risk Metrics

| Asset | VaR 95% | Annualized Sharpe Ratio |
| ----- | ------- | ----------------------- |
| TSLA  | 0.0547  | 0.7782                  |
| BND   | 0.0049  | 0.3568                  |
| SPY   | 0.0172  | 0.7939                  |

## 5. Key Insights

- TSLA has delivered strong returns historically but with significantly higher volatility and drawdown risk.
- BND provides stability and diversification benefits but at the cost of lower returns.
- SPY offers balanced exposure with moderate volatility and relatively high Sharpe Ratio.
- VaR and Sharpe Ratios confirm the trade-off between risk and return across asset classes.

# Next Steps

Based on the findings in Task 1, the following steps are recommended for the next phases of the project:

1. **Model Development (Task 2)** – Implement ARIMA/SARIMA and LSTM models for TSLA price forecasting, ensuring chronological train-test splits. Perform hyperparameter tuning for optimal performance.
2. **Forecast Analysis (Task 3)** – Use the best-performing model to generate 6–12 month forecasts for TSLA, including confidence intervals and trend/volatility analysis.
3. **Portfolio Optimization (Task 4)** – Incorporate forecasted TSLA returns and historical BND/SPY returns into a Modern Portfolio Theory (MPT) framework using PyPortfolioOpt to identify the efficient frontier, maximum Sharpe ratio, and minimum volatility portfolios.
4. **Backtesting (Task 5)** – Simulate the optimized strategy over the final year of available data and compare its performance against a benchmark portfolio (e.g., 60% SPY / 40% BND).
5. **Final Report** – Prepare a professional investment memo including methodology, findings, visualizations, and final recommendations for the GMF Investment Committee.
