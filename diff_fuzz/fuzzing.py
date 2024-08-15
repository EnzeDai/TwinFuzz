import os
import time
import configparser
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

# Load configurations from config.ini
def read_conf():
    conf = configparser.ConfigParser()
    conf.read('config.ini')
    name = conf.get('model', 'name')
    dataset = conf.get('model', 'dataset')

    return name, dataset

# Load models for inference
name, dataset = read_conf()
resist_model = keras.models.load_model("")
vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}.h5")


# filter seeds?

 
 



lr = 0.1
sample_set = []

start = time.time()
# Start fuzzing

