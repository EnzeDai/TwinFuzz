import os
import time
import tensorflow as tf
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load for inference
resist_model = keras.models.load_model("")
vulner_model = keras.models.load_model("")


# filter seeds?

 
 



lr = 0.1
data_sets = []

start = time.time()
# Start fuzzing

