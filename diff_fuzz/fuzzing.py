import os
import sys
import time
import configparser
import numpy as np
import tensorflow as tf
from tensorflow import keras
sys.path.append("../")
from attacks import deepfool

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

ATTACK_SAMPLE_LIMIT = 10000
ATTACK_SAMPLE_PATH = 'DeepFool_Atks.npz'

# Load configurations from config.ini
def read_conf():
    conf = configparser.ConfigParser()
    conf.read('config.ini')
    name = conf.get('model', 'name')
    dataset = conf.get('model', 'dataset')
    adv_sample_num = conf.get('model', 'advSample')

    return name, dataset, adv_sample_num

# DeepFool attack generator
def df_atk_loader():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    adv_all = []
    for img in x_train:
        _, _, orig_label, adv_label, adv_img = deepfool.deepfool(img, vulner_model)
        if adv_label != orig_label:
            adv_all.append(adv_img)
        if len(adv_all) % 1000 == 0:
            print("[INFO] Now Successful DeepFool Attack Num:", len(adv_all))
            if len(adv_all) == ATTACK_SAMPLE_LIMIT: break

    print("[INFO] Success DeepFool Attack Num:", len(adv_all))
    adv_all = tf.Variable(adv_all).numpy() # shape = (limit, 1, 28, 28)
    adv_all = adv_all.reshape(adv_all.shape[0], 28, 28, 1)
    np.savez('./DeepFool_Atks.npz', advs=adv_all)
    
    return adv_all
    

if __name__ == "__main__":

    # Load models for inference
    name, dataset, adv_sample_num = read_conf()
    resist_model = keras.models.load_model(f"../{dataset}/checkpoint/{name}_{dataset}_Adv_{adv_sample_num}.h5")
    vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}.h5")

    # Attack side samples generation
    if os.path.exists(ATTACK_SAMPLE_PATH):
        print('[INFO]: Adversarial samples have been generated.')
        with np.load(ATTACK_SAMPLE_PATH) as f:
            afv_all = f['advs']
    else:
        adv_all = df_atk_loader()


    # differential testing
    seeds_filter = []

    resist_pred_idxs = np.argmax(resist_model(adv_all), axis=1)
    vulner_pred_idxs = np.argmax(vulner_model(adv_all), axis=1)

    print(resist_pred_idxs)

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

