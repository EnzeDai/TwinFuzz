import os
import sys
import time
import configparser
import numpy as np
import tensorflow as tf
from tensorflow import keras

import consts
sys.path.append("../")
from attacks import deepfool
from seed_ops import filter_data

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

# Find same predictions
def find_same(preds1, preds2):
    same_preds = [] # format = (index, value)

    if len(preds1) == len(preds2):
        for i in range(len(preds1)):
            if preds1[i] == preds2[i]:
                same_preds.append((i, preds1[i]))
    
    return same_preds

# DeepFool attack generator
def df_atk_loader(model):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    adv_all = []
    for img in x_train:
        _, _, orig_label, adv_label, adv_img = deepfool.deepfool(img, model)
        if adv_label != orig_label:
            adv_all.append(adv_img)
        if len(adv_all) % 1000 == 0:
            print("[INFO] Now Successful DeepFool Attack Num:", len(adv_all))
            if len(adv_all) == consts.ATTACK_SAMPLE_LIMIT: break

    print("[INFO] Success DeepFool Attack Num:", len(adv_all))
    adv_all = tf.Variable(adv_all).numpy() # shape = (limit, 1, 28, 28)
    adv_all = adv_all.reshape(adv_all.shape[0], 28, 28, 1)
    np.savez(consts.ATTACK_SAMPLE_PATH, advs=adv_all)
    
    return adv_all
    

if __name__ == "__main__":

    # Load models for inference
    name, dataset, adv_sample_num = read_conf()
    resist_model = keras.models.load_model(f"../{dataset}/checkpoint/{name}_{dataset}_Adv_{adv_sample_num}.h5")
    vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}.h5")

    # Attack side samples generation
    if os.path.exists(consts.ATTACK_SAMPLE_PATH):
        print('[INFO]: Adversarial samples have been generated.')
        with np.load(consts.ATTACK_SAMPLE_PATH) as f:
            adv_all = f['advs']
    else:
        adv_all = df_atk_loader(model=vulner_model)


    # differential testing
    resist_pred_idxs = np.argmax(resist_model(adv_all), axis=1)
    vulner_pred_idxs = np.argmax(vulner_model(adv_all), axis=1)

    # print(resist_pred_idxs)
    # print(vulner_pred_idxs)

    # Filter
    filter_idxs = filter_data(consts.ATTACK_SAMPLE_PATH)
    with np.load(consts.FILTER_SAMPLE_PATH) as f:
        adv_filt = f['advf']
    
    resist_pred_idxs = np.argmax(resist_model(adv_filt), axis=1)
    vulner_pred_idxs = np.argmax(vulner_model(adv_filt), axis=1)

    # print(filter_idxs)
    # print(resist_pred_idxs)
    # print(vulner_pred_idxs)

    # Find same predicitons
    same_preds = find_same(resist_pred_idxs, vulner_pred_idxs)
    print(same_preds)


    lr = 0.1
    sample_set = []

    start = time.time()
    # Start fuzzing
    # for idx in adv_filt:
    #     delta_t = time.time() - start
    #     # Limit time
    #     if delta_t > 300:
    #         break
        
    #     img_list = []

