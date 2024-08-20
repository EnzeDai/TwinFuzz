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
    adv_sample_num = conf.get('model', 'advSample')

    return name, dataset, adv_sample_num

# Load models for inference
name, dataset, adv_sample_num = read_conf()
resist_model = keras.models.load_model(f"../{dataset}/checkpoint/{name}_{dataset}_Adv_{adv_sample_num}.h5")
vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}.h5")


# differential testing
seeds_filter = []
'''
    resist_predicts = resist_model.predict(input_data)
    vulner_predicts = vulner_model.predict(input_data)
    # check difference
'''



lr = 0.1
sample_set = []

start = time.time()
# Start fuzzing
for idx in seeds_filter:
    delta_t = time.time() - start
    # Limit time
    if delta_t > 300:
        break
    
    img_list = []
    

    
