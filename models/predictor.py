# models/predictor.py
import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
# Set the random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
class SequencePredictor:
    def __init__(self, input_dim, seq_length=3):
        self.seq_length = seq_length
        
        # Create a more complex LSTM model
        model = Sequential([
            LSTM(128, input_shape=(seq_length, input_dim), return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64),
            BatchNormalization(),
            Dropout(0.2),
            Dense(input_dim)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        
    def create_sequences(self, data, seq_length=None):
        """Create input sequences and target outputs"""
        if seq_length is None:
            seq_length = self.seq_length
            
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    def train(self, X, y=None, epochs=20, batch_size=32, validation_split=0.1):
        # If y is not provided, create sequences from X
        if y is None:
            X, y = self.create_sequences(X)
            
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True)
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return history
    
    def detect_anomalies(self, X, y=None, threshold_factor=2.5):
        # If y is not provided, create sequences from X
        if y is None:
            if len(X.shape) == 2:  # If X is not already in sequences
                X, y = self.create_sequences(X)
            else:
                # We need y for comparison, return error if not available
                raise ValueError("For sequence prediction, target values (y) are required for anomaly detection")
        
        # Predict next sequence elements
        y_pred = self.model.predict(X)
        
        # Calculate prediction error (MSE)
        mse = np.mean(np.square(y - y_pred), axis=1)
        
        # Dynamic threshold based on error distribution
        threshold = np.mean(mse) + threshold_factor * np.std(mse)
        
        # Identify anomalies
        anomalies = mse > threshold
        
        return anomalies, mse
