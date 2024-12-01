import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_files(dataset_path):
    """Lists and loads CSV files from a local directory."""
    if os.path.exists(dataset_path):
        files_in_directory = os.listdir(dataset_path)
        print("Files in directory:", files_in_directory)
        file_paths = glob.glob(os.path.join(dataset_path, "*.csv"))
        return file_paths
    else:
        print(f"The directory does not exist: {dataset_path}")
        return []

def preprocess_data(file_paths):
    """Preprocesses and combines data from multiple files."""
    dfs = []
    for file_path in file_paths:
        if 'stock_metadata' not in file_path and 'NIFTY50_all' not in file_path:
            try:
                # Read the CSV file
                df = pd.read_csv(file_path, parse_dates=['Date'])
                # Check if the DataFrame is empty
                if df.empty:
                    print(f"File {file_path} is empty. Skipping.")
                    continue
                # Check for essential columns
                required_columns = ['Date', 'Close']
                if not all(col in df.columns for col in required_columns):
                    print(f"File {file_path} is missing required columns. Skipping.")
                    continue
                # Add stock name as a new column
                df['Stock'] = os.path.basename(file_path).replace('.csv', '')
                dfs.append(df)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
    # Filter out empty DataFrames and concatenate
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.sort_index(inplace=True)
        return combined_df
    else:
        print("No valid data to combine.")
        return pd.DataFrame()  # Return an empty DataFrame


def feature_engineering(df):
    """Adds various technical indicators and performs data preprocessing."""
    df['Daily Return'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Daily Return'].rolling(window=30).std()
    df.dropna(inplace=True)
    return df

def plot_closing_price(df):
    """Visualizes the closing price over time."""
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x=df.index, y='Close', label='Closing Price', color='blue')
    plt.title('Closing Price Over Time')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculates MACD, Signal Line, and Histogram."""
    df['EMA_12'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df

def plot_macd(df):
    """Plots the MACD and Signal Line."""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=2)
    plt.plot(df.index, df['Signal_Line'], label='Signal Line', color='red', linewidth=2)
    plt.bar(df.index, df['MACD_Histogram'], label='MACD Histogram', color='grey', alpha=0.5)
    plt.title('MACD and Signal Line')
    plt.legend(loc='upper left')
    plt.show()

def normalize_data(df):
    """Normalizes continuous variables and applies transformations."""
    scaler = StandardScaler()
    df[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])
    minmax_scaler = MinMaxScaler()
    df[['Volume']] = minmax_scaler.fit_transform(df[['Volume']])
    df['Log_Volume'] = np.log1p(df['Volume'])
    return df

def main():
    # Specify the local dataset path
    dataset_path = r"d:\Studies\4th Year\Data Mining\Project\new\data"  # Update this path

    # Load dataset
    file_paths = load_files(dataset_path)
    if not file_paths:
        print("No valid files to process. Exiting.")
        return
    
    # Process data
    df = preprocess_data(file_paths)
    if df.empty:
        print("No valid data found in the files. Exiting.")
        return

    df = feature_engineering(df)
    df = normalize_data(df)
    df = calculate_macd(df)

    # Plot data
    plot_closing_price(df)
    plot_macd(df)


if __name__ == '__main__':
    main()
