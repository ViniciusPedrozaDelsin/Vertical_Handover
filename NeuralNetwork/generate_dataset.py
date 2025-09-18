import numpy as np
import pandas as pd

# Dataset Name
dataset_name = "RMSE_10inps"

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
N = 100_000

def custom_random_array(samples, zero_chance, one_chance):
    r = np.random.rand(samples)
    values = np.random.rand(samples)
    
    values[r < zero_chance] = 0
    values[r > (1-one_chance)] = 1
    
    return values

data = {
    "RSSI": custom_random_array(N, 0.25, 0.25),
    "SNR": custom_random_array(N, 0.25, 0.25),
    "Throughput": custom_random_array(N, 0.25, 0.25),
    "PC": custom_random_array(N, 0.25, 0.25),
    "MC": custom_random_array(N, 0.25, 0.25),
    "BER": custom_random_array(N, 0.25, 0.25),
    "FEC": custom_random_array(N, 0.25, 0.25),
    "Delay": custom_random_array(N, 0.35, 0.35),
    "Jitter": custom_random_array(N, 0.35, 0.35),
    "HC": np.random.randint(0, 2, N),
}

df = pd.DataFrame(data)

# Compute the output
df["Output"] = ((df["RSSI"]**2 + df["SNR"]**2 + df["Throughput"]**2 + df["PC"]**2 + df["MC"]**2 + df["BER"]**2 + df["FEC"]**2 + df["Delay"]**2 + df["Jitter"]**2 + df["HC"]**2)/10)**(1/2)

# Reorder columns
df = df[['RSSI', 'SNR', 'Throughput', 'PC', 'MC', 'BER', 'FEC', 'Delay', 'Jitter', 'HC', 'Output']]

# Save to CSV
df.to_csv(f"{dataset_name}.csv", index=False)

print(f"Dataset created and saved as {dataset_name}.csv")
print(df.head())
