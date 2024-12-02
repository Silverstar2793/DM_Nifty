import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU

warnings.filterwarnings("ignore")

def process_csv(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df = df[['Date', 'Close']]
    
    # Plot stock prices
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label="Stock Prices")
    plt.title(f"Stock Prices Over Time ({os.path.basename(file_path)})")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.savefig(f"{os.path.splitext(os.path.basename(file_path))[0]}_StockPrice.jpg")
    print(f"Saved stock prices plot for {file_path}.")

    # Prepare training and testing data
    dataset = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    training_data_len = int(len(scaled_data) * 0.9)
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - 60:]

    # Create sequences for training and testing
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(seq_length, len(data)):
            x.append(data[i - seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    seq_length = 60
    x_train, y_train = create_sequences(train_data, seq_length)
    x_test, y_test = create_sequences(test_data, seq_length)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build GRU model
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=64, epochs=10)

    # Predict
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Compare with actual data
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'][training_data_len:], y_test_scaled, label="Actual Prices")
    plt.plot(df['Date'][training_data_len:], predictions, label="Predicted Prices")
    plt.title(f"Stock Price Prediction ({os.path.basename(file_path)})")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.savefig(f"{os.path.splitext(os.path.basename(file_path))[0]}_Prediction.jpg")
    print(f"Saved prediction plot for {file_path}.")

    # Print model summary and return predictions
    model.summary()
    return predictions, y_test_scaled

def process_folder(folder_path):
    # Iterate over all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_path}")
            predictions, actual = process_csv(file_path)
            print("\nSample Predictions vs Actual Values:")
            for pred, act in zip(predictions[:5], actual[:5]):
                print(f"Predicted: {pred[0]:.2f}, Actual: {act[0]:.2f}")

if __name__ == "__main__":
    folder_path = "archive"  # Replace with the path to your folder containing CSV files
    process_folder(folder_path)
