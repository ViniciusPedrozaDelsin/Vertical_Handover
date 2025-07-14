import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Fields: RSSI, SNR, Throughput, BER, FEC, PC, MC
model = tf.keras.models.load_model("TOPSIS_NN_OUTPUT.keras")

plot_model(model, to_file="test.png", show_shapes=True)