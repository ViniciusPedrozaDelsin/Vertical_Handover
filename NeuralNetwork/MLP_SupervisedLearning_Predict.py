import tensorflow as tf
import numpy as np

# Fields: RSSI, SNR, Throughput, BER, FEC, PC, MC
model = tf.keras.models.load_model("TOPSIS_NN_50_200_50.keras")

# Example: [[0.008149, 0.027735, 0.000000, 1.000000, 0.846581, 0.00, 0.00]]
X_test = [[0.00000000e+00, 0.00000000e+00, 1.93798588e-01, 0.00000000e+00, 5.90092349e-05, 2.00000000e-01, 1.00000000e+00]]

X_test = np.array(X_test)

y_pred = model.predict(X_test)

print(y_pred)