import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('./data/TOPSIS.csv', index_col=False)
print(df.shape)     # (rows, columns)
print(df.columns)   # List of column names
print(df.dtypes)    # Data types of each column

# Load dataset
iris = load_iris()
X = iris.data
y_raw = iris.target
y = y_raw.reshape(-1, 1)  # reshape for OneHotEncoder

print("Input features shape:", X.shape)
print(f"Labels before reshape: {y_raw}")
print("Labels before encoding:", np.unique(y))

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nExample of normalized features (first 3 samples):")
print(X_scaled[:3])

# One-hot encode target labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

print("\nExample of one-hot encoded labels (first 5 samples):")
print(y_encoded[:5])
x=input("")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

print("\nTraining set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Build the model
model = Sequential([
    Dense(2, input_shape=(X.shape[1],), activation='relu'),
    Dense(2, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    verbose=1,
    validation_split=0.1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🔍 Final Evaluation on Test Data:")
print(f"Loss (categorical crossentropy): {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Make predictions
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\n🔎 Predictions vs Actual:")
for pred, actual in zip(y_pred_classes, y_true_classes):
    print(f"Predicted: {pred} | Actual: {actual}")