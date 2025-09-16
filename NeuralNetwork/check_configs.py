import tensorflow as tf

print("TF version:", tf.__version__)

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)