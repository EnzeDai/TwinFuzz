import os
import sys
import time
import random
import configparser
import numpy as np
import tensorflow as tf
from tensorflow import keras

import consts
sys.path.append("../")
from attacks import deepfool, mim_atk, bim_atk
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
    

# Momentum_Iterative_Method Attack DataLoader for MNIST:
def mim_atk_loader(model):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    adv_all = []
    atk_numbers=0
    for imgs, label in zip(x_train, y_train):
        
        # Normalize the input image
        test_img = np.expand_dims(imgs,axis=-1).astype(np.float32)/ 255
        # Load the normal model
        normal_model = model
        # Load the logits model
        logits_model = model
        test_image = test_img
        test_label = label
        # start attacking:
        print("Attacked Image Number")
        # Output the adv_image
        adv_image = mim_atk.momentum_iterative_method(
            model_fn=logits_model,
            x=tf.convert_to_tensor(np.expand_dims(test_image, axis=0)), 
            eps=0.4,
            eps_iter=0.08,
            nb_iter=20,
            norm=np.inf,
            clip_min=0.0,
            clip_max=1.0,
            y=tf.convert_to_tensor([test_label]),  
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



# Basic_Iterative_Method Attack DataLoader for MNIST:

def bim_atk_loader(model):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    adv_all = []
    atk_numbers=0
    for imgs, label in zip(x_train, y_train):
        
        # Normalize the input image
        test_img = np.expand_dims(imgs,axis=-1).astype(np.float32)/ 255
        # Load the normal model
        normal_model = model
        # Load the logits model
        logits_model = model
        test_image = test_img
        test_label = label
        # start attacking:
        print("Attacked Image Number")
        # Output the adv_image
        # Run the BIM attack
        adv_image = bim_atk.basic_iterative_method(
            model_fn=model,
            x=tf.convert_to_tensor(np.expand_dims(test_image, axis=0)),  # Add batch dimension
            # x = test_image,
            eps=0.4,
            eps_iter=0.08,
            nb_iter=20,
            norm=np.inf,
            clip_min=0.0,
            clip_max=1.0,
            y=tf.convert_to_tensor([test_label]),  
            #y = test_label,
            targeted=False,
            rand_init=False,
            rand_minmax=0.3, #Initialize
            sanity_checks=False,
        )
        # Calculate the total perturbation
        tot_pert = np.linalg.norm(adv_image - test_image)
        # Predict the original label
        orig_label = np.argmax(normal_model.predict(np.expand_dims(test_image, axis = 0)))
        # Predict the adv label after the adversarial attack
        adv_label = np.argmax(normal_model.predict(adv_image))

        # Release the redundant dimension:
        adv_image = tf.squeeze(adv_image, axis =-1) 

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
    np.savez(consts.ATTCK_SAMPLE_PATH_BIM, advs=adv_all)
    return adv_all



if __name__ == "__main__":

    # Load models for inference
    name, dataset, adv_sample_num = read_conf()
    resist_model = keras.models.load_model(f"../{dataset}/checkpoint/{name}_{dataset}_Adv_{adv_sample_num}.h5")
    vulner_model = keras.models.load_model(f"../{dataset}/{name}_{dataset}.h5")

    # Attack side samples generation
    if os.path.exists(consts.ATTCK_SAMPLE_PATH_BIM):
        print('[INFO]: Adversarial samples have been generated.')
        with np.load(consts.ATTCK_SAMPLE_PATH_BIM) as f:
            adv_all = f['advs']
    else:
        adv_all = bim_atk_loader(model=vulner_model)


    # differential testing
    resist_pred_idxs = np.argmax(resist_model(adv_all), axis=1)
    vulner_pred_idxs = np.argmax(vulner_model(adv_all), axis=1)

    # print(resist_pred_idxs)
    # print(vulner_pred_idxs)

    # Filter
    filter_idxs = filter_data(consts.ATTCK_SAMPLE_PATH_BIM)
    with np.load(consts.FILTER_SAMPLE_PATH_BIM) as f:
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
    for idx, pred in same_preds:
        delta_t = time.time() - start
        # Limit time
        if delta_t > 300:
            break
        
        img_list = []
        tmp_img = adv_filt[idx]
        tmp_img = np.expand_dims(tmp_img, axis=0) # (28, 28, 1) => (1, 28, 28, 1)

        orig_img = tmp_img.copy()
        orig_norm = np.linalg.norm(orig_img) # L2 Norm
        img_list.append(tf.identity(tmp_img))

        # Predictions
        softmax = resist_model(tmp_img)
        orig_index = np.argmax(softmax[0])
        one_hot = keras.utils.to_categorical([orig_index], 10)
        label_top5 = np.argsort(softmax[0][:-5])

        folMax = 0
        epoch = 0
        total_sets = []
        while len(img_list) > 0:
            gen_img = img_list.pop(0)
            for _ in range(2):
                gen_img = tf.Variable(gen_img)
                with tf.GradientTape(persistent=True) as tape:
                    loss = keras.losses.categorical_crossentropy(one_hot, resist_model(gen_img))
                    grads = tape.gradient(loss, gen_img)
                    fol = tf.norm(grads+1e-20)
                    tape.watch(fol)
                    softmax = resist_model(gen_img)
                    obj = fol - softmax[0][orig_index]
                    dl_di = tape.gradient(obj, gen_img)  # minimize obj

                del tape

                gen_img = gen_img + dl_di * lr * (random.random() + 0.5)
                gen_img = tf.clip_by_value(gen_img, clip_value_min=0, clip_value_max=1)

                with tf.GradientTape() as tape:
                    tape.watch(gen_img)
                    loss = keras.losses.categorical_crossentropy(one_hot, resist_model(gen_img))
                    grads = tape.gradient(loss, gen_img)
                    fol = np.linalg.norm(grads.numpy())

                # make sure perturbation is not too large
                dist = np.linalg.norm(gen_img.numpy() - orig_img) / orig_norm
                print(dist)
                if fol > folMax and dist < consts.PERTURBATION_LIMIT:
                    folMax = fol
                    img_list.append(tf.identity(gen_img))
                
                gen_index = np.argmax(resist_model(gen_img)[0])
                if gen_index != orig_index:
                    total_sets.append((fol, gen_img.numpy()))

                print('FOL:', fol, 'FOL max:', folMax)
                