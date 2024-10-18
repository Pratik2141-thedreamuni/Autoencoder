import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for data generation
n_data_points = 15000
time_series = pd.date_range(start='2024-10-01', periods=n_data_points, freq='S')  # 1 second intervals

# Baseline temperature (simulate normal operating temperature around 30°C)
base_temp = 30

# Simulate long-term trend (temperature slowly rising due to machine wear-and-tear)
long_term_trend = np.linspace(base_temp, base_temp + 1.5, n_data_points)

# Simulate periodic cycles (e.g., machine heating up and cooling down daily)
cycle_period = 3600  # 1 hour cycle
cycles = 0.5 * np.sin(np.arange(n_data_points) * 2 * np.pi / cycle_period)

# Inject random noise
noise = np.random.normal(0, 0.2, n_data_points)

# Combine all components to create synthetic temperature data
normal_temperature = long_term_trend + cycles + noise

# Inject anomalies (spikes to simulate machine overheating or cooling failure)
anomalies = normal_temperature.copy()
anomaly_indices = np.random.choice(np.arange(n_data_points), size=50, replace=False)
anomalies[anomaly_indices] += np.random.uniform(3, 5, 50)  # Add random spike of 3 to 5 degrees

# Create DataFrame
temperature_data = pd.DataFrame({
    'time': time_series,
    'temperature': anomalies
})

# Save to CSV
temperature_data.to_csv('synthetic_temperature_data.csv', index=False)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(temperature_data['time'], temperature_data['temperature'], label='Temperature with Anomalies')
plt.title('Synthetic Temperature Data with Anomalies (15000 points)')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
