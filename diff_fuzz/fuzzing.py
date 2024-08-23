import os
import sys
import time
import configparser
import tensorflow as tf
from tensorflow import keras
sys.path.append("../")
from attacks import cw_2_atk, cw_inf_atk

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


if __name__ == "__main__":

    # Load models for inference
    name, dataset, adv_sample_num = read_conf()
    resist_model = keras.models.load_model(f"../{dataset}/checkpoint/{name}_{dataset}_Adv_{adv_sample_num}.h5")
    vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}.h5")

    # Attack side sample generate
    cw_l2 = cw_2_atk.CwL2(vulner_model)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Reshape
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # One-Hot Label
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    adv = cw_l2.attack(x_train, y_train)

    print(len(adv))


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
    

    
