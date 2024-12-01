import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

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
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_layer_size, output_size):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
                self.fc = nn.Linear(hidden_layer_size, output_size)

            def forward(self, x):
                lstm_out, (h_n, c_n) = self.lstm(x)
                predictions = self.fc(lstm_out[:, -1])
                return predictions

        self.model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
        print("LSTM model built successfully.")

    def train_model(self, epochs=20, batch_size=32):
        """Train the LSTM model."""
        sequence_length = 60
        X_train, y_train = self.create_sequences(self.train_data, sequence_length)
        self.build_model((X_train.shape[1], 1))
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.model(X_batch.unsqueeze(-1))
                loss = criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        print("Model trained successfully.")

    def evaluate_model(self):
        """Evaluate the model and visualize predictions."""
        sequence_length = 60
        X_test, y_test = self.create_sequences(self.test_data, sequence_length)
        
        # Convert to PyTorch tensors
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test.unsqueeze(-1)).squeeze().numpy()

        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
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
