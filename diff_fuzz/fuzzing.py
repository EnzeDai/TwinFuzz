import os
import sys
import time
import configparser
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
# from tensorflow.keras.utils import to_categorical

import consts
sys.path.append("../")
from attacks import deepfool
from seed_ops import filter_data
from attacks import mim_atk, utils_attack
# from attacks.utils_attack import optimize_linear, compute_gradient, clip_eta

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

# Other Attack method generator:
def cw_atk_loader(model, model_logits):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    adv_all = []
    for imgs in x_train:
        target_class_ohe = to_categorical(0, num_classes=10)
        attack = cw_test.CW2(model_logits, k = 0)
        _, _, orig_label, adv_label, adv_img = attack.generate(model, imgs, target_class_ohe)
        if adv_label != orig_label:
            adv_all.append(adv_img)
        if len(adv_all) % 1000 == 0:
            print("[INFO] Now Successful CW Attack Num:", len(adv_all))
            if len(adv_all) == consts.ATTACK_SAMPLE_LIMIT: break

    print("[INFO] Success CW Attack Num:", len(adv_all))
    adv_all = tf.Variable(adv_all).numpy() # shape = (limit, 1, 28, 28)
    adv_all = adv_all.reshape(adv_all.shape[0], 28, 28, 1)
    np.savez(consts.ATTCK_SAMPLE_PATH_CW2, advs=adv_all)


def mim_atk_loader(model, model_logits):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    adv_all = []
    atk_numbers=0
    for imgs, label in zip(x_train, y_train):
        
        # Normalize the input image
        test_img = np.expand_dims(imgs,axis=-1).astype(np.float32)/ 255
        # Load the normal model
        normal_model = model
        # Load the logits model
        logits_model = model_logits
        test_image = test_img
        test_label = label
        # start attacking:
        print("Attacked Image Number")
        
        adv_image = mim_atk.momentum_iterative_method(
            model_fn=logits_model,
            x=tf.convert_to_tensor(np.expand_dims(test_image, axis=0)),  # Add batch dimension
            eps=0.4,
            eps_iter=0.08,
            nb_iter=20,
            norm=np.inf,
            clip_min=0.0,
            clip_max=1.0,
            y=tf.convert_to_tensor([test_label]),  # True label, change if you want a targeted attack
            targeted=False,
            decay_factor=1.0,
            sanity_checks=False,
        )
        tot_pert = np.linalg.norm(adv_image - test_image)
        # Predict the original label
        orig_label = np.argmax(normal_model.predict(np.expand_dims(test_image, axis = 0)))
        # Predict the adv label after the adversarial attack
        adv_label = np.argmax(normal_model.predict(adv_image))
        # Output results
        print(f"Original Label: {orig_label}")
        print(f"Adversarial Label: {adv_label}")
        print(f"Total Perturbation (L2 norm): {tot_pert}")
        # print(f"Total Iterations: {10}")  # The number of iterations is specified by `nb_iter`
        atk_numbers = atk_numbers + 1
        print(atk_numbers)
        if adv_label != orig_label:
            adv_all.append(adv_image)
        if len(adv_all) % 1000 == 0:
            print("[INFO] Now Successful MIM Attack Num:", len(adv_all))
            if len(adv_all) == consts.ATTACK_SAMPLE_LIMIT: break

    print("[INFO] Success MIM Attack Num:", len(adv_all))
    adv_all = tf.Variable(adv_all).numpy() # shape = (limit, 1, 28, 28)
    adv_all = adv_all.reshape(adv_all.shape[0], 28, 28, 1)
    np.savez(consts.ATTCK_SAMPLE_PATH_MIM, advs=adv_all)
    return adv_all
        
    

if __name__ == "__main__":

    # Load models for inference
    name, dataset, adv_sample_num = read_conf()
    resist_model = keras.models.load_model(f"../{dataset}/checkpoint/{name}_{dataset}_Adv_{adv_sample_num}.h5")
    vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}_normal.h5")
    vulner_logits_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}_logits.h5")



    # Attack side samples generation
    if os.path.exists(consts.ATTCK_SAMPLE_PATH_MIM):
        print('[INFO]: Adversarial samples have been generated.')
        with np.load(consts.ATTCK_SAMPLE_PATH_MIM) as f:
            adv_all = f['advs']
    else:
        print('[INFO]: Now Generating Adversarial Samples.')
        adv_all = mim_atk_loader(model=vulner_model, model_logits = vulner_logits_model)
        # print(adv_all)
        # adv_all = cw_atk_loader(model=vulner_model, model_logits = vulner_logits_model)


    # differential testing
    resist_pred_idxs = np.argmax(resist_model(adv_all), axis=1)
    vulner_pred_idxs = np.argmax(vulner_model(adv_all), axis=1)

    print(resist_pred_idxs)
    print(vulner_pred_idxs)

    # Filter
    filter_data(consts.ATTCK_SAMPLE_PATH_MIM)
    with np.load(consts.FILTER_SAMPLE_PATH_MIM) as f:
        adv_filt = f['advf']
    
    resist_pred_idxs = np.argmax(resist_model(adv_filt), axis=1)
    vulner_pred_idxs = np.argmax(vulner_model(adv_filt), axis=1)

    print(resist_pred_idxs)
    print(vulner_pred_idxs)


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

