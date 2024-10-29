import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Flatten
from tensorflow.keras.optimizers import Adam


# Simulate fetching Twitter mentions data
def fetch_twitter_mentions(start_date, end_date):
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    return np.random.randint(0, 100, size=num_days).reshape(-1, 1)


# Align two dataframes by their index (date) and fill missing values
def align_data(stock_df, mentions_df):
    # Ensure both dataframes have a DateTime index
    stock_df.index = pd.to_datetime(stock_df.index)
    mentions_df.index = pd.to_datetime(mentions_df.index)

    # Align both dataframes by their index
    combined_df = pd.concat([stock_df, mentions_df], axis=1, join='inner')
    return combined_df


# Step 1: Data Download and Preprocessing
def download_and_prepare_multivariate_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Fetch Twitter mentions data
    mentions_data = fetch_twitter_mentions(start_date, end_date)
    mentions_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Check if length of dates and data match
    if len(mentions_dates) != len(mentions_data):
        raise ValueError("Mismatch between the number of dates and the number of mentions data points.")

    mentions_df = pd.DataFrame(mentions_data, index=mentions_dates, columns=['TweetMentions'])

    # Align data
    combined_df = align_data(stock_data, mentions_df)

    features = combined_df[['Open', 'High', 'Low', 'Close', 'Volume', 'TweetMentions']].values

    # Scale data to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)

    X, y = [], []
    sequence_length = 60  # Use 60 days of data to predict the next day's price
    future_days = 10  # Predict the next 10 days

    for i in range(sequence_length, len(scaled_data) - future_days):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i:i + future_days, 3])  # Predict future Close prices

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# Step 2: Model Architectures
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(10))  # Predict 10 future steps
    return model


def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(10))
    return model


def build_ffn_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))  # Predict 10 future steps
    return model


# Step 3: Compile and Train the Models
def compile_and_train(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, learning_rate=0.001):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return model, history


# Step 4: Combine Predictions (Ensemble)
def ensemble_predictions(models, X_test):
    predictions = [model.predict(X_test) for model in models]
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction


# Step 5: Evaluate the Ensemble Model
def evaluate_ensemble_model(models, X_test, y_test, scaler):
    predictions = ensemble_predictions(models, X_test)

    # Inverse transform only the relevant columns (the predicted Close prices)
    zeros = np.zeros((predictions.shape[0], scaler.n_features_in_))

    # Insert the predictions back into the zero-filled array
    zeros[:, 3] = predictions[:, 0]
    predictions = scaler.inverse_transform(zeros)[:, 3]

    # Insert the y_test back into the zero-filled array
    zeros[:, 3] = y_test[:, 0]
    y_test = scaler.inverse_transform(zeros)[:, 3]

    mse = mean_squared_error(y_test, predictions)
    print(f'Ensemble Model Mean Squared Error: {mse}')

    # Visualize the prediction vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:10], label='True Price')
    plt.plot(predictions[:10], label='Ensemble Predicted Price')
    plt.legend()
    plt.show()


# Main Function to Run the Experiment
def main():
    ticker = 'AMZN'
    start_date = '2018-01-01'
    end_date = '2023-01-01'

    X, y, scaler = download_and_prepare_multivariate_data(ticker, start_date, end_date)

    # Split into training and validation sets
    split_ratio = 0.8
    train_size = int(len(X) * split_ratio)
    X_train, X_val, y_train, y_val = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Build and train different models
    lstm_model = build_lstm_model(X_train.shape[1:])
    gru_model = build_gru_model(X_train.shape[1:])
    ffn_model = build_ffn_model(X_train.shape[1:])

    lstm_model, _ = compile_and_train(lstm_model, X_train, y_train, X_val, y_val)
    gru_model, _ = compile_and_train(gru_model, X_train, y_train, X_val, y_val)
    ffn_model, _ = compile_and_train(ffn_model, X_train, y_train, X_val, y_val)

    # Combine models into an ensemble
    models = [lstm_model, gru_model, ffn_model]

    # Evaluate the ensemble model
    evaluate_ensemble_model(models, X_val, y_val, scaler)


if __name__ == "__main__":
    main()
