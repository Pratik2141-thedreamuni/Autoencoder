import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

def build_autoencoder(input_dim):
    # Input Layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder with more layers and batch normalization
    encoder = Dense(256, activation="relu", activity_regularizer=regularizers.l2(10e-5))(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.3)(encoder)
    
    encoder = Dense(128, activation="relu")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.3)(encoder)
    
    encoder = Dense(64, activation="relu")(encoder)
    encoder = BatchNormalization()(encoder)
    
    # Latent Space
    latent_space = Dense(32, activation="relu")(encoder)
    
    # Decoder (mirroring the encoder structure)
    decoder = Dense(64, activation="relu")(latent_space)
    decoder = BatchNormalization()(decoder)
    
    decoder = Dense(128, activation="relu")(decoder)
    decoder = BatchNormalization()(decoder)
    
    decoder = Dense(256, activation="relu")(decoder)
    
    # Output Layer
    output_layer = Dense(input_dim, activation='sigmoid')(decoder)
    
    # Create the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    
    return autoencoder

# Train the Autoencoder with model checkpoint to save weights
def train_autoencoder(X_train, X_test):
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)
    autoencoder.summary()

    checkpoint = ModelCheckpoint('best_autoencoder_weights.keras', save_best_only=True, monitor='val_loss', mode='min')
    history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=64, 
                              validation_data=(X_test, X_test), 
                              callbacks=[checkpoint], verbose=2)
    
    return autoencoder, history

# Evaluate the Autoencoder Model
def evaluate_model(autoencoder, X_test):
    X_test_pred = autoencoder.predict(X_test)
    reconstruction_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    threshold = np.mean(reconstruction_loss) + 3 * np.std(reconstruction_loss)
    print(f'Set anomaly threshold: {threshold}')

    anomalies = np.sum(reconstruction_loss > threshold)

    # Handle divide-by-zero by masking the zero values in X_test
    mask = X_test != 0
    avg_deviation_percent = np.mean(np.abs((X_test_pred[mask] - X_test[mask]) / X_test[mask])) * 100

    return reconstruction_loss, threshold, anomalies, avg_deviation_percent
