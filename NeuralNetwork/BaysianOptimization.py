import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import random
import sys

# Counter
trial_counter = 0

# Redirect stdout to a file
log_file = open("bayesian_optimization_log.txt", "w")
sys.stdout = log_file

# Load and preprocess data
df = pd.read_csv('./data/TOPSIS.csv', index_col=False)
df = df.drop('Hash', axis=1)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

print("============================================")

# Objective function for Optuna
def objective(trial):
    
    print("============================================")
    
    global trial_counter
    print(f"Trial Number: {trial_counter}")
    
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers", 1, 4)
    units = []
    for _ in range(n_layers):
        #unit = trial.suggest_int("units", 2, 8)
        unit = random.randint(2, 8)
        units.append(unit)
    print(f"Neural Network Shape: {units}")
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    print(f"Learning Rate: {learning_rate}")
    
    n_epochs = trial.suggest_int("n_epochs", 20, 40)
    print(f"Number of epochs: {n_epochs}")
    
    n_batch_size = random.choice([2, 4, 8, 16, 32, 64, 128])
    print(f"Batch size: {n_batch_size}")

    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(units[0], activation='relu'))
    i = 1
    for _ in range(n_layers - 1):
        model.add(Dense(units[i], activation='relu'))
        i += 1
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    # Stop the model early if the validation loss start to increase
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=n_epochs,
        batch_size=n_batch_size,
        verbose=0,
        callbacks=[early_stop]
    )

    val_loss = history.history['val_loss'][-1]
    print(f"Validation Loss: {val_loss}")
    
    trial_counter += 1
    
    print("============================================")
    
    return val_loss

# Run Bayesian Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

print("\nBest hyperparameters:")
print(study.best_params)

# Close the log file
log_file.close()

# Restore stdout
sys.stdout = sys.__stdout__