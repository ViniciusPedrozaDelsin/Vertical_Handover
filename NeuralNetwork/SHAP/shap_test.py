import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================
# 1. Generate Dummy QoS Data
# ==========================================
np.random.seed(42)
n_samples = 2000

data = pd.DataFrame({
    "RSSI": np.random.uniform(-100, -40, n_samples),
    "SNR": np.random.uniform(0, 40, n_samples),
    "Throughput": np.random.uniform(1, 500, n_samples),
    "PC": np.random.uniform(0, 10, n_samples),
    "MC": np.random.uniform(0, 10, n_samples),
    "BER": np.random.uniform(0, 0.01, n_samples),
    "FEC": np.random.uniform(0.3, 0.9, n_samples),
    "Delay": np.random.uniform(1, 100, n_samples),
    "Jitter": np.random.uniform(0, 50, n_samples),
    "HC": np.random.uniform(0, 5, n_samples)
})

# ==========================================
# 2. Create a Fake "Network Score"
# ==========================================
score = (
    0.35 * data["Throughput"]
    + 0.25 * data["RSSI"]
    + 0.15 * data["SNR"]
    - 0.15 * data["Delay"]
    - 0.05 * data["BER"]
)
y = score.values
X = data.values
feature_names = data.columns

# ==========================================
# 3. Train/Test Split & Scaling
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 4. Build Neural Network
# ==========================================
model = Sequential([
    Dense(32, activation="relu", input_shape=(10,)),
    Dense(16, activation="relu"),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

# ==========================================
# 5. SHAP Analysis
# ==========================================
explainer = shap.Explainer(model, X_train[:100])
shap_values = explainer(X_test[:20])  # use a subset for speed

# ==========================================
# 6. SHAP Global Importance Plots
# ==========================================
shap.summary_plot(shap_values.values, X_test[:20], feature_names=feature_names)
shap.summary_plot(shap_values.values, X_test[:20], feature_names=feature_names, plot_type="bar")

# ==========================================
# 7. Single-Sample Force Plot
# ==========================================
sample_index = 0
shap.force_plot(
    shap_values.base_values[sample_index],  # <-- use base_values instead of explainer.expected_value
    shap_values.values[sample_index],
    X_test[sample_index],
    feature_names=feature_names,
    matplotlib=True
)
plt.show()

print("SHAP analysis complete!")