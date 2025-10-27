# Gold Price Time Series Analysis and Forecasting

## Overview

This project focuses on analyzing historical daily gold (XAUUSD) closing prices and volume data and applying various time series forecasting models to predict future price movements. The analysis includes exploring data characteristics, testing for stationarity, differencing the series, and implementing ARIMA, SARIMA, and Exponential Smoothing models. The project concludes by evaluating the performance of these models and recommending the best-performing one based on key metrics.

## Objective / Problem Statement

The primary objective is to build and evaluate time series models for forecasting the daily closing price of gold (XAUUSD). This addresses the common problem in financial markets of predicting asset prices, which is crucial for trading strategies, risk management, and investment decisions. The project aims to demonstrate the application of classical time series techniques to a real-world financial dataset.

## Data Sources

The dataset used is a historical daily price and volume data for XAUUSD, loaded from the file `/content/drive/MyDrive/Colab Notebooks/Predict Gold Price/XAU_1d_data.csv`. The data includes 'date', 'open', 'high', 'low', 'close', and 'volume' columns.

## Methodology and Tools Used

The project follows a standard time series analysis and forecasting workflow:

1.  **Data Loading and Preprocessing**: Loading data from a CSV file using Pandas, converting the date column to datetime objects, setting the date as the index, and inspecting for missing values.
2.  **Exploratory Data Analysis (EDA)**: Visualizing the closing price and volume over time to observe trends and patterns.
3.  **Time Series Decomposition**: Decomposing the series into trend, seasonal, and residual components using `seasonal_decompose` to understand the underlying structures.
4.  **Stationarity Testing**: Performing the Augmented Dickey-Fuller (ADF) test using `adfuller` to check for stationarity, a critical assumption for many time series models.
5.  **Differencing**: Applying first-order differencing using `.diff()` to the non-stationary series to achieve stationarity, and re-testing with ADF.
6.  **ACF and PACF Analysis**: Plotting the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) using `plot_acf` and `plot_pacf` on the differenced series to identify potential parameters for ARIMA-family models.
7.  **Model Implementation**:
    *   Implementing an ARIMA model with parameters suggested by ACF/PACF analysis.
    *   Implementing a SARIMA model to account for potential seasonality.
    *   Implementing an Exponential Smoothing model (Holt's Linear Trend) as an alternative approach.
8.  **Model Evaluation**: Calculating key regression metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) using `sklearn.metrics` to quantify model performance on a test set.
9.  **Model Comparison**: Presenting the evaluation metrics in a comparison table and visualizing the actual vs. predicted prices for each model.

**Technical Stack**:

*   Python
*   Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, sklearn

## Key Results and Insights

*   The XAUUSD close price series exhibits a clear upward trend and is non-stationary, as confirmed by the ADF test (p-value > 0.05).
*   First-order differencing successfully transformed the series into a stationary one (ADF p-value ≈ 0.0000).
*   ACF and PACF analysis on the differenced series suggested an AR(1) process, indicating an ARIMA(1,1,0) model as a potential fit.
*   Upon evaluation, the **Exponential Smoothing (Holt's Linear Trend) model demonstrated the best performance** among the implemented models, exhibiting significantly lower RMSE (418.35) and MAE (251.80) compared to the ARIMA (RMSE 582.34, MAE 376.31) and SARIMA (RMSE 583.02, MAE 377.11) models.
*   The negative R2 values for ARIMA and SARIMA (-0.63 and -0.63 respectively) further highlight their poor fit to the test data, contrasting with the positive R2 (0.16) of the Exponential Smoothing model.
*   Visualizations confirmed that the Exponential Smoothing model's predictions followed the general trend of the actual prices in the test set more closely than the ARIMA/SARIMA models, which tended to produce flatter forecasts.

## Skills Demonstrated

*   Time Series Analysis
*   Financial Data Analysis
*   Data Loading and Preprocessing (Pandas)
*   Exploratory Data Analysis (EDA)
*   Data Visualization (Matplotlib, Seaborn)
*   Stationarity Testing (ADF Test)
*   Series Differencing
*   ACF and PACF Analysis
*   ARIMA, SARIMA, and Exponential Smoothing Modeling
*   Model Fitting and Forecasting
*   Model Evaluation (MSE, RMSE, MAE, R2)
*   Interpretation of Time Series Diagnostics

## Business / Research Value

This project demonstrates a practical application of time series forecasting techniques to a financial asset. The ability to analyze historical price data, identify underlying patterns (trend, seasonality), and build predictive models is highly valuable in finance for:

*   **Algorithmic Trading**: Developing strategies based on price predictions.
*   **Risk Management**: Forecasting potential price volatility.
*   **Investment Analysis**: Informing decisions on asset allocation and timing.
*   **Benchmarking**: Comparing the performance of different forecasting approaches.

The project showcases the technical rigor required for quantitative roles and the ability to translate theoretical concepts into actionable code for financial analysis.

## How to Run the Notebook

1.  Clone this repository.
2.  Ensure you have a Jupyter Notebook environment (e.g., Google Colab, Anaconda).
3.  Install the required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `sklearn`. (A `requirements.txt` file could be added for easier installation).
4.  Place the `XAU_1d_data.csv` file in the specified path (`/content/drive/MyDrive/Colab Notebooks/Predict Gold Price/XAU_1d_data.csv`) or update the file path in the notebook.
5.  Run the cells sequentially in the Jupyter Notebook.

## Future Work or Extensions

*   **Hyperparameter Tuning**: Systematically tune the parameters of the Exponential Smoothing model and potentially explore auto-ARIMA for automated parameter selection.
*   **Incorporate External Factors**: Include macroeconomic indicators, geopolitical events, or sentiment data as exogenous variables in models like SARIMAX or machine learning models.
*   **Advanced Models**: Experiment with more sophisticated models such as Prophet, LSTM, or GARCH (for volatility modeling).
*   **Walk-Forward Validation**: Implement a walk-forward testing approach for a more robust evaluation of model performance over time.
*   **Confidence Intervals**: Generate and visualize prediction intervals to understand the uncertainty of the forecasts.
*   **Explore Different Frequencies**: Analyze and forecast gold prices at different time frequencies (e.g., weekly, monthly).
