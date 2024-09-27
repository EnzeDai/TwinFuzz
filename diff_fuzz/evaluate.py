import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

import consts
import fuzzing
sys.path.append("../")
from attacks import deepfool, mim_atk, bim_atk

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Two models under testing
name, dataset, adv_sample_num, _ = fuzzing.read_conf()
resist_model = keras.models.load_model(f"../{dataset}/checkpoint/{name}_{dataset}_Adv_{adv_sample_num}.h5")
vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}.h5")
# Model after testing
enhance_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}_DiffEntro.h5")

# Load eval inputs for Robustness Evaluation
eval_files = [consts.DF_EVAL_PATH, consts.MIM_EVAL_PATH, consts.BIM_EVAL_PATH]
for file in eval_files:
    if os.path.exists(file):
        pass
    else:
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        eval_all = []
        for img in x_test:
            _, _, orig_label, adv_label, adv_img = deepfool.deepfool(img, vulner_model)
            if adv_label != orig_label:
                eval_all.append(adv_img)
            
        print("[INFO] Success DeepFool Attack Num:", len(eval_all))
        eval_all = tf.Variable(eval_all).numpy()
        eval_all = eval_all.reshape(eval_all.shape[0], 28, 28, 1)
        np.savez(consts.DF_EVAL_PATH, evals=eval_all)

sNums = [600*i for i in [8, 12, 16, 20]]

for num in sNums:
    pass

# Data selection
