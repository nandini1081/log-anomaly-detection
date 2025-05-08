# models/autoencoder.py
import numpy as np
import tensorflow as tf
import random
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout

# Set the random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

class AutoencoderModel:
    def __init__(self, input_dim, latent_dim=32):
        # Define architecture with LSTM layers
        inputs = Input(shape=(1, input_dim))
        
        # Encoder
        encoded = LSTM(64, return_sequences=True)(inputs)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(latent_dim)(encoded)
        
        # Decoder
        decoded = RepeatVector(1)(encoded)
        decoded = LSTM(64, return_sequences=True)(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = TimeDistributed(Dense(input_dim))(decoded)
        
        # Autoencoder model
        self.autoencoder = Model(inputs, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        # Create separate encoder model for later use
        self.encoder = Model(inputs, encoded)
    
    def train(self, X, epochs=20, batch_size=32, validation_split=0.1):
        # Reshape X for LSTM input: (samples, time steps, features)
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Train with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = self.autoencoder.fit(
            X_reshaped, X_reshaped,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return history
    
    def detect_anomalies(self, X, threshold_factor=2.0):
        # Reshape X for LSTM input
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        
        # Reconstruct input data
        X_pred = self.autoencoder.predict(X_reshaped)
        
        # Calculate reconstruction error (MSE) for each sample
        mse = np.mean(np.square(X_reshaped - X_pred), axis=(1, 2))
        
        # Determine threshold dynamically based on distribution of errors
        threshold = np.mean(mse) + threshold_factor * np.std(mse)
        
        # Identify anomalies
        anomalies = mse > threshold
        
        return anomalies, mse


