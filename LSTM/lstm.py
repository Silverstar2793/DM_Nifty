import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

class StockLSTMPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_data = None
        self.test_data = None
        self.model = None

    def load_data(self):
        """Load and preprocess stock data."""
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df[['Date', 'Close']]
        self.df['Close'] = self.scaler.fit_transform(self.df[['Close']])
        print("Data loaded and normalized successfully.")

    def split_data(self, sequence_length=60):
        """Split the data into training and testing sets."""
        split_ratio = 0.9
        data = self.df['Close'].values
        train_size = int(len(data) * split_ratio)
        self.train_data = data[:train_size]
        self.test_data = data[train_size - sequence_length:]
        print("Data split into training and testing sets.")

    def create_sequences(self, data, sequence_length=60):
        """Create sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build the LSTM model."""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        print("LSTM model built successfully.")

    def train_model(self, epochs=20, batch_size=32):
        """Train the LSTM model."""
        sequence_length = 60
        X_train, y_train = self.create_sequences(self.train_data, sequence_length)
        self.build_model((X_train.shape[1], 1))
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        print("Model trained successfully.")

    def evaluate_model(self):
        """Evaluate the model and visualize predictions."""
        sequence_length = 60
        X_test, y_test = self.create_sequences(self.test_data, sequence_length)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Date'][-len(y_test):], y_test, label='Actual Prices', color='blue')
        plt.plot(self.df['Date'][-len(predictions):], predictions, label='Predicted Prices', color='red', linestyle='dashed')
        plt.title('LSTM Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

    def smape(self, y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error."""
        return np.mean((np.abs(y_pred - y_true) * 200) / (np.abs(y_pred) + np.abs(y_true)))

# Example usage:
if __name__ == '__main__':
    lstm_predictor = StockLSTMPredictor(data_path='path_to_your_csv_file.csv')
    lstm_predictor.load_data()
    lstm_predictor.split_data()
    lstm_predictor.train_model(epochs=20, batch_size=32)
    lstm_predictor.evaluate_model()
