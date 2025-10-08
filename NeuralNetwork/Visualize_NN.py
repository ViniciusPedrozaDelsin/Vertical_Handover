import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Fields: RSSI, SNR, Throughput, BER, FEC, PC, MC
model = tf.keras.models.load_model("RMSE_RL_17inps_VELOCITY_EMBEDDING_W10_G09_32_64_32_16.keras")

plot_model(model, to_file="test.png", show_shapes=True)