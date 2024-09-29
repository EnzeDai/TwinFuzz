import os
import sys
import configparser
import numpy as np
import tensorflow as tf
from tensorflow import keras

import consts
import loader
import fuzzing
sys.path.append("../")

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices("GPU")
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
    test_duration = conf.get('model', 'duration')

    return name, dataset, adv_sample_num, test_duration

# Two models under testing
name, dataset, adv_sample_num, _ = read_conf()
resist_model = keras.models.load_model(f"../{dataset}/checkpoint/{name}_{dataset}_Adv_{adv_sample_num}.h5")
vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}.h5")
# Model after testing
enhance_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}_DiffEntro.h5")

# Prepare eval inputs for Robustness Evaluation
if not os.path.exists(consts.DF_EVAL_PATH):
    loader.df_eval_loader(vulner_model)
elif not os.path.exists(consts.MIM_EVAL_PATH):
    loader.mim_eval_loader(vulner_model)
elif not os.path.exists(consts.BIM_EVAL_PATH):
    loader.bim_eval_loader(vulner_model)

# Load eval inputs for Robustness Evaluation
# DeepFool
with np.load(consts.DF_EVAL_PATH) as df:
    df_test, df_labels = df['eval'], df['labels']

resist_eval_idxs = np.argmax(resist_model(df_test), axis=1)
same_preds = fuzzing.find_same(resist_eval_idxs, df_labels)
rob_acc_df = len(same_preds) / len(df_test)

# MIM
with np.load(consts.MIM_EVAL_PATH) as mi:
    mi_test, mi_labels = mi['eval'], mi['labels']

resist_eval_idxs = np.argmax(resist_model(mi_test), axis=1)
same_preds = fuzzing.find_same(resist_eval_idxs, mi_labels)
rob_acc_mi = len(same_preds) / len(mi_test)

# BIM
with np.load(consts.BIM_EVAL_PATH) as bi:
    bi_test, bi_labels = bi['eval'], bi['labels']

resist_eval_idxs = np.argmax(resist_model(bi_test), axis=1)
same_preds = fuzzing.find_same(resist_eval_idxs, bi_labels)
rob_acc_bi = len(same_preds) / len(bi_test)

print("DeepFool Rob Acc:", rob_acc_df, "MIM Rob Acc:", rob_acc_mi, "BIM Rob Acc:", rob_acc_bi)