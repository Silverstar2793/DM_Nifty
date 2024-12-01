import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

class StockPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.predictions = None

    def load_data(self):
        """Load and preprocess stock data."""
        self.df = pd.read_csv(self.data_path)
        self.df.replace(np.nan, inplace=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m-%d')
        self.df = self.df[['Date', 'Close']]
        print("Data loaded successfully.")
    
    def visualize_data(self):
        """Visualize the stock data."""
        plt.figure(figsize=(14, 7))
        plt.title('Stock Prices')
        plt.xlabel('Dates')
        plt.ylabel('Close Prices')
        plt.plot(self.df['Date'], self.df['Close'])
        plt.show()
    
    def split_data(self):
        """Split the data into training and testing sets."""
        split_ratio = 0.9
        self.train_data = self.df[0:int(len(self.df) * split_ratio)]
        self.test_data = self.df[int(len(self.df) * split_ratio):]
        print("Data split into training and testing sets.")
    
    def check_stationarity(self):
        """Check if the series is stationary."""
        result = adfuller(self.df['Close'].dropna())
        print(f"ADF Statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        print("Critical Values:")
        for key, value in result[4].items():
            print(f"   {key}: {value}")
    
    def plot_acf_pacf(self):
        """Plot ACF and PACF."""
        diff = self.df['Close'].diff().dropna()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        ax1.plot(diff)
        ax1.set_title("First Difference")
        plot_acf(diff, ax=ax2)
        plt.show()
        plot_pacf(diff, lags=10)
        plt.show()
    
    def train_arima(self, order=(2, 1, 1)):
        """Train ARIMA model."""
        train_values = self.train_data['Close'].values
        test_values = self.test_data['Close'].values
        history = [x for x in train_values]
        self.predictions = []
        
        for t in range(len(test_values)):
            self.model = ARIMA(history, order=order)
            model_fit = self.model.fit()
            yhat = model_fit.forecast()[0]
            self.predictions.append(yhat)
            history.append(test_values[t])
        
        error = mean_squared_error(test_values, self.predictions)
        print(f"Mean Squared Error: {error:.3f}")
        return model_fit.summary()
    
    def visualize_predictions(self):
        """Visualize predictions vs actual values."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Date'], self.df['Close'], label='Original Data', color='blue')
        plt.plot(self.test_data['Date'], self.predictions, label='Predicted Data', linestyle='dashed', color='green')
        plt.plot(self.test_data['Date'], self.test_data['Close'], label='Actual Data', color='red')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    
    def smape(self, y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error."""
        return np.mean((np.abs(y_pred - y_true) * 200) / (np.abs(y_pred) + np.abs(y_true)))

# Example usage:
if __name__ == '__main__':
    predictor = StockPredictor(data_path='path_to_your_csv_file.csv')
    predictor.load_data()
    predictor.visualize_data()
    predictor.split_data()
    predictor.check_stationarity()
    predictor.plot_acf_pacf()
    model_summary = predictor.train_arima(order=(2, 1, 1))
    print(model_summary)
    predictor.visualize_predictions()
