# models/classifier.py
import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
# Set the random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

class ClassifierModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.scaler = StandardScaler()
        
        # LSTM-based classifier for anomaly detection
        model = Sequential([
            LSTM(128, input_shape=(1, input_dim), return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Binary classification problem (normal vs anomaly)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
    def fit(self, X, anomaly_ratio=0.1, epochs=20, batch_size=32):
        """Train the classifier with synthetic anomalies"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Reshape for LSTM input
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Create synthetic labels (assuming most training data is normal)
        # This is a semi-supervised approach where we assume some percentage of data is anomalous
        reconstruction_errors = self._compute_reconstruction_errors(X_reshaped)
        threshold = np.percentile(reconstruction_errors, 100 * (1 - anomaly_ratio))
        y = (reconstruction_errors > threshold).astype(int)
        
        # Train the model
        history = self.model.fit(
            X_reshaped, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ],
            verbose=1
        )
        
        return history
    
    def _compute_reconstruction_errors(self, X):
        """Compute reconstruction errors for semi-supervised labeling"""
        # Simple autoencoder for reconstruction error
        temp_model = Sequential([
            LSTM(32, input_shape=(1, self.input_dim), return_sequences=True),
            LSTM(16),
            Dense(32, activation='relu'),
            Dense(self.input_dim)
        ])
        temp_model.compile(optimizer='adam', loss='mse')
        
        # Train a simple model to get reconstruction errors
        temp_model.fit(X, X.reshape(X.shape[0], self.input_dim), 
                       epochs=5, batch_size=32, verbose=0)
        
        # Calculate reconstruction error
        X_pred = temp_model.predict(X)
        errors = np.mean(np.square(X.reshape(X.shape[0], self.input_dim) - X_pred), axis=1)
        
        return errors
    
    def predict(self, X):
        """Predict anomalies in test data"""
        # Scale and reshape
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Get anomaly probabilities
        anomaly_probs = self.model.predict(X_reshaped).flatten()
        
        # Convert to binary predictions with default threshold of 0.5
        anomalies = anomaly_probs > 0.5
        
        return anomalies