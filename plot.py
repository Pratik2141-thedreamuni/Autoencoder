import matplotlib.pyplot as plt

# Function to plot training and validation loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

# Function to plot reconstruction loss distribution
def plot_reconstruction_loss(reconstruction_loss):
    plt.hist(reconstruction_loss, bins=50)
    plt.title('Reconstruction Loss Distribution')
    plt.xlabel('Reconstruction Loss')
    plt.ylabel('Frequency')
    plt.show()
