# data_loading_processing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load the synthetic temperature data
    data = pd.read_csv('synthetic_temperature_data.csv')
    temperature = data['temperature'].values.reshape(-1, 1)

    # Normalize the data (scaling between 0 and 1)
    scaler = MinMaxScaler()
    temperature_scaled = scaler.fit_transform(temperature)

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test = train_test_split(temperature_scaled, test_size=0.2, random_state=42)

    return X_train, X_test
