from data_loading_processing import load_and_preprocess_data
from evaluate import train_autoencoder, evaluate_model

# Load and preprocess the data
X_train, X_test = load_and_preprocess_data()

# Train the autoencoder
autoencoder, history = train_autoencoder(X_train, X_test)

# Evaluate the autoencoder model
reconstruction_loss, threshold, anomalies, avg_deviation_percent = evaluate_model(autoencoder, X_test)

# Print anomalies and average deviation percentage
print(f"Number of anomalies detected: {anomalies}")
print(f"Average Deviation Percentage: {avg_deviation_percent:.2f}%")

# Save the trained model weights for future use (fixing the extension issue)
autoencoder.save_weights('final_autoencoder_weights.weights.h5')
