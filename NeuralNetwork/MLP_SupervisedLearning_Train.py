import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('./data/TOPSIS.csv', index_col=False)
df = df.drop('Hash', axis=1)

# All columns except the last
df_features = df.iloc[:, :-1]

# Just the last column
df_target = df.iloc[:, -1]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(7, input_shape=(df_features.shape[1],), activation='relu'),
    Dense(6, activation='relu'),
    Dense(2, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.002901195099910745), loss='mean_squared_error', metrics=['mae'])

# Stop the model early if the validation loss start to increase
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=34,
    batch_size=16,
    verbose=1,
    validation_split=0.1,
    callbacks=[early_stop]
)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Evaluation on Test Data:")
print(f"Loss (mean_squared_error): {loss:.4f}")
print(f"MAE: {mae:.4f}")
print("====================")
print(X_test)
print("====================")

# Predict
y_pred = model.predict(X_test)

print("Predictions vs Actual:")
for pred, actual in zip(y_pred.flatten(), y_test.to_numpy().flatten()):
    print(f"Predicted: {pred:.4f} | Actual: {actual:.4f}")

model.save("TOPSIS_NN_7_6_2.keras")
