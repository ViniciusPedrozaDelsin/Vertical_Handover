import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import random

# Load and preprocess data
df = pd.read_csv('./data/TOPSIS.csv', index_col=False)
df = df.drop('Hash', axis=1)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 4)
    units = []
    for _ in range(n_layers):
        #unit = trial.suggest_int("units", 2, 8)
        unit = random.randint(2, 8)
        units.append(unit)
    print(units)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_epochs = trial.suggest_int("n_epochs", 20, 40)

    model = Sequential()
    model.add(Dense(units[0], input_shape=(X.shape[1],), activation='relu'))
    i = 1
    for _ in range(n_layers - 1):
        model.add(Dense(units[i], activation='relu'))
        i += 1
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=n_epochs,
        batch_size=8,
        verbose=0
    )

    val_loss = history.history['val_loss'][-1]
    return val_loss

# Run Bayesian Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=150)

print("\nBest hyperparameters:")
print(study.best_params)
