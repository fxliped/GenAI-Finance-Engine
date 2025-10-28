"""
train_lstm.py

Trains LSTM models for stock price prediction based on historical hourly data
and aggregated daily sentiment scores.

Workflow:
1. Loads 2-year hourly stock data (.csv) and daily sentiment data (.csv) for each ticker.
2. Merges the datasets, forward-filling sentiment scores.
3. Selects features ('Close', 'Volume', 'sentiment_score') and scales them (0-1).
4. Creates sequences of past data (e.g., 60 hours) to predict the next hour's closing price.
5. Defines a PyTorch LSTM model architecture.
6. Defines a custom PyTorch Dataset for sequences.
7. Splits data into training and testing sets.
8. Trains the LSTM model using PyTorch DataLoader, MSELoss, and Adam optimizer.
9. Evaluates the model on the test set using Root Mean Squared Error (RMSE).
10. Saves the trained model's state dictionary (.pth) and the target scaler (.pkl)
    for each ticker in the ./models/ directory.

Assumes data exists in ./stock_data/ and ./sentiment_data/.
Outputs models and scalers to ./models/.
"""

import os
import sys
import time
import pickle
from datetime import timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#############################################
# --- Configuration ---
#############################################

TICKERS_TO_PROCESS = ["AAPL", "META", "NVDA", "GOOGL", "MSFT",
                      "AMZN", "AVGO", "JPM", "NFLX", "ORCL"]
STOCK_DATA_PATH = "stock_data"
SENTIMENT_DATA_PATH = "sentiment_data"
MODEL_SAVE_PATH = "models" # Directory to save trained models

# Data Preparation Parameters
SEQUENCE_LENGTH = 60        # Past hours of data to use for prediction
PREDICTION_HORIZON = 1      # How many hours ahead to predict (e.g., 1 hour)
FEATURES = ['Close', 'Volume', 'sentiment_score'] # Input features for model
TARGET_COLUMN = 'Close'     # Aim to predict closing value

# Model Parameters
INPUT_SIZE = len(FEATURES)
HIDDEN_SIZE = 64        # Number of LSTM units in each layer
NUM_LAYERS = 2          # Number of LSTM layers stacked
OUTPUT_SIZE = 1         # Predicting a single value (Close price)

# Training Hyperparameters
NUM_EPOCHS = 20         # How many times to loop through the training data
BATCH_SIZE = 64         # Number of sequences per training batch
LEARNING_RATE = 0.001   # Step size for optimizer
TEST_SPLIT = 0.2        # Use 20% of data for testing

#############################################
# --- Data Preparation ---
#############################################

def load_and_prepare_data(ticker):
    """
    Loads hourly stock data and daily sentiment data for a ticker, merges them,
    scales the features and target, and creates input/output sequences for LSTM.

    Args:
        ticker (str): The stock ticker symbol.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None, MinMaxScaler | None, MinMaxScaler | None]:
            - X: NumPy array of input sequences (shape: [num_samples, seq_length, num_features]).
            - y: NumPy array of target values (shape: [num_samples]).
            - feature_scaler: Fitted MinMaxScaler for input features.
            - target_scaler: Fitted MinMaxScaler for the target variable ('Close').
            Returns (None, None, None, None) if data loading or processing fails.
    """
    print(f"Loading data for {ticker}...")
    try:
        stock_file = os.path.join(STOCK_DATA_PATH, f"{ticker}_2y_hourly.csv")
        sentiment_file = os.path.join(SENTIMENT_DATA_PATH, f"{ticker}_sentiment.csv")

        # Load data
        stock_df = pd.read_csv(stock_file)
        sentiment_df = pd.read_csv(sentiment_file)

        # Convert date columns to datetime objects
        stock_df['Datetime'] = pd.to_datetime(stock_df['Datetime'], utc=True).dt.tz_localize(None)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

        # Set index for easier merging
        stock_df.set_index('Datetime', inplace=True)
        # Create a date column from the stock index for merging
        stock_df['date'] = stock_df.index.date
        stock_df['date'] = pd.to_datetime(stock_df['date'])


        # Merge sentiment data (forward-fill to match hourly stock data)
        merged_df = pd.merge_ordered(stock_df.reset_index(), 
                                     sentiment_df, 
                                     on='date', 
                                     fill_method='ffill')
        merged_df.set_index('Datetime', inplace=True)
        merged_df.dropna(inplace=True) # Drop rows if sentiment couldn't be forward-filled

        if merged_df.empty:
            raise ValueError("Merged DataFrame is empty after processing.")

        # Select features and scale them
        data_to_scale = merged_df[FEATURES].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_to_scale)

        # Also scale the target separately for inverse transform later
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(merged_df[[TARGET_COLUMN]])


        # Create sequences
        X, y = [], []
        target_col_index = FEATURES.index(TARGET_COLUMN) # Find index of 'Close'
        for i in range(len(scaled_data) - SEQUENCE_LENGTH - PREDICTION_HORIZON + 1):
            X.append(scaled_data[i:(i + SEQUENCE_LENGTH)])
            # Target is the 'Close' price PREDICTION_HORIZON steps ahead
            y.append(scaled_data[i + SEQUENCE_LENGTH + PREDICTION_HORIZON - 1, target_col_index])

        X, y = np.array(X), np.array(y)

        print(f"Data loaded. Shape X: {X.shape}, Shape y: {y.shape}")
        return X, y, target_scaler # Return scaler for inverse transform later

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return None, None, None

#############################################
# --- PYTORCH DATASET ---
#############################################

class StockDataset(Dataset):
    """
    Custom PyTorch Dataset to wrap stock data sequences and targets.
    Converts NumPy arrays to PyTorch tensors.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Ensure target is 2D tensor [num_samples, 1] for loss calculation
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#############################################
# --- LSTM MODEL DEFINITION ---
#############################################

class LSTMModel(nn.Module):
    """
    Defines the LSTM network architecture.
    Consists of LSTM layers followed by a fully connected layer for output.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Number of LSTM units in each layer.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Dimension of the output (usually 1 for price prediction).
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

#############################################
# --- TRAINING FUNCTION ---
#############################################

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    Trains the provided PyTorch model.

    Args:
        model (nn.Module): The LSTM model instance.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): The loss function (e.g., MSELoss).
        optimizer (torch.optim.Optimizer): The optimization algorithm (e.g., Adam).
        num_epochs (int): Number of epochs to train for.
        device (torch.device): The device (CPU or CUDA) to train on.
    """
    print("\nStarting training...")
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass: Compute predicted outputs by passing inputs to the model
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}')
    print("Training finished.")

#############################################
# --- MAIN EXECUTION ---
#############################################

if __name__ == "__main__":
    # Ensure model save directory exists
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    for ticker in TICKERS_TO_PROCESS:
        print(f"\n==================== TRAINING MODEL FOR {ticker} ====================")
        # Load and Prepare Data
        X, y, target_scaler = load_and_prepare_data(ticker)

        if X is None or y is None:
            print("Exiting due to data loading errors.")
            exit()

        # Split data into Training and Testing sets
        split_index = int(len(X) * (1 - TEST_SPLIT))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Create Datasets and DataLoaders
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # No shuffle for test loader - order matters for plotting time series
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize Model, Loss, Optimizer
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train the Model
        train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, device)

        # Evaluate the Model 
        print("\nEvaluating model on test data...")
        model.eval() # Set model to evaluation mode
        all_predictions = []
        all_actuals = []
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                all_predictions.extend(outputs.cpu().numpy())
                all_actuals.extend(labels.cpu().numpy())

        # Inverse transform predictions and actuals to original scale
        predictions_rescaled = target_scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))
        actuals_rescaled = target_scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1))

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(actuals_rescaled, predictions_rescaled))
        print(f'Test Root Mean Squared Error (RMSE): {rmse:.4f}')


    # Save the Trained Model
        model_filename = os.path.join(MODEL_SAVE_PATH, f"{ticker}_lstm_model.pth")
        torch.save(model.state_dict(), model_filename)
        print(f"\n✅ Model saved to {model_filename}")

        # Also save the target scaler, needed to convert future predictions back to actual prices
        scaler_filename = os.path.join(MODEL_SAVE_PATH, f"{ticker}_target_scaler.pkl")
        import pickle
        with open(scaler_filename, 'wb') as f:
            pickle.dump(target_scaler, f)
        print(f"✅ Target scaler saved to {scaler_filename}")

    print("\n Finished training models for all tickers!")