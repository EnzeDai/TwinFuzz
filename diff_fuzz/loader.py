import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

import consts
import fuzzing
sys.path.append("../")
from attacks import deepfool, mim_atk, bim_atk

name, dataset, _, test_time = fuzzing.read_conf()

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

# DeepFool evaluation generator
def df_eval_loader(model):
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    eval_all = []
    labels = []
    for img in x_test:
        _, _, orig_label, adv_label, adv_img = deepfool.deepfool(img, model)
        if adv_label != orig_label:
            eval_all.append(adv_img)
            labels.append(orig_label)
        if len(eval_all) % 1000 == 0:
            print("[INFO] Now Successful DeepFool Evalution Num:", len(eval_all))

    print("[INFO] Success DeepFool Evaluation Num:", len(eval_all))
    eval_all = tf.Variable(eval_all).numpy() # shape = (limit, 1, 28, 28)
    eval_all = eval_all.reshape(eval_all.shape[0], 28, 28, 1)
    np.savez(consts.DF_EVAL_PATH.split('.')[0] + name + dataset + '.npz', eval=eval_all, labels=labels)
    
    return eval_all, labels

# Momentum_Iterative_Method Attack DataLoader for MNIST:
def mim_atk_loader(model):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    adv_all = []
    atk_numbers=0
    for imgs, label in zip(x_train, y_train):
        
        # Normalize the input image
        test_img = np.expand_dims(imgs,axis=-1).astype(np.float32)/ 255
        normal_model = model
        logits_model = model
        test_image = test_img
        test_label = label
        # start attacking:
        print("Attacked Image Number")
        # Output the adv_image
        adv_image = mim_atk.momentum_iterative_method(model_fn=logits_model, x=tf.convert_to_tensor(np.expand_dims(test_image, axis=0)), 
            eps=0.4, eps_iter=0.08, nb_iter=20, norm=np.inf, clip_min=0.0, clip_max=1.0, y=tf.convert_to_tensor([test_label]),  
            targeted=False, decay_factor=1.0, sanity_checks=False,
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

# Momentum_Iterative_Method evaluation DataLoader for MNIST:
def mim_eval_loader(model):
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    eval_all = []
    labels = []
    atk_numbers=0
    for imgs, label in zip(x_test, y_test):
        
        # Normalize the input image
        test_img = np.expand_dims(imgs,axis=-1).astype(np.float32)/ 255
        normal_model = model
        logits_model = model
        test_image = test_img
        test_label = label
        # start attacking:
        print("Attacked Image Number")
        # Output the adv_image
        adv_image = mim_atk.momentum_iterative_method(model_fn=logits_model, x=tf.convert_to_tensor(np.expand_dims(test_image, axis=0)), 
            eps=0.4, eps_iter=0.08, nb_iter=20, norm=np.inf, clip_min=0.0, clip_max=1.0, y=tf.convert_to_tensor([test_label]),  
            targeted=False, decay_factor=1.0, sanity_checks=False,
        )
        tot_pert = np.linalg.norm(adv_image - test_image)
        # Predict the original label
        orig_label = np.argmax(normal_model.predict(np.expand_dims(test_image, axis = 0)))
        # Predict the adv label after the adversarial attack
        adv_label = np.argmax(normal_model.predict(adv_image))
        # Output results
        # print(f"Original Label: {orig_label}, Adversarial Label: {adv_label}, Total Perturbation (L2 norm): {tot_pert}")
        # print(f"Total Iterations: {10}")  # The number of iterations is specified by `nb_iter`
        atk_numbers = atk_numbers + 1
        print(atk_numbers)
        if adv_label != orig_label:
            eval_all.append(adv_image)
            labels.append(orig_label)
        if len(eval_all) % 1000 == 0:
            print("[INFO] Now Successful MIM Evaluation Num:", len(eval_all))
    
    print("[INFO] Success MIM Evaluation Num:", len(eval_all))
    eval_all = tf.Variable(eval_all).numpy() # shape = (limit, 1, 28, 28)
    eval_all = eval_all.reshape(eval_all.shape[0], 28, 28, 1)
    np.savez(consts.MIM_EVAL_PATH.split('.')[0] + name + dataset + '.npz', eval=eval_all, labels=labels)
    
    return eval_all, labels

# Basic_Iterative_Method Attack DataLoader for MNIST:
def bim_atk_loader(model):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    adv_all = []
    atk_numbers=0
    for imgs, label in zip(x_train, y_train):
        
        # Normalize the input image
        test_img = np.expand_dims(imgs,axis=-1).astype(np.float32)/ 255
        normal_model = model
        logits_model = model
        test_image = test_img
        test_label = label
        # start attacking:
        print("Attacked Image Number")
        # Output the adv_image
        # Run the BIM attack
        adv_image = bim_atk.basic_iterative_method(model_fn=model, x=tf.convert_to_tensor(np.expand_dims(test_image, axis=0)),  # Add batch dimension
            eps=0.4, eps_iter=0.08, nb_iter=20, norm=np.inf, clip_min=0.0, clip_max=1.0, y=tf.convert_to_tensor([test_label]),
            targeted=False, rand_init=False, rand_minmax=0.3, #Initialize
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

# Basic_Iterative_Method Evaluation DataLoader for MNIST:
def bim_eval_loader(model):
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    eval_all = []
    labels = []
    atk_numbers=0
    for imgs, label in zip(x_test, y_test):
        
        # Normalize the input image
        test_img = np.expand_dims(imgs,axis=-1).astype(np.float32)/ 255
        normal_model = model
        logits_model = model
        test_image = test_img
        test_label = label
        # start attacking:
        print("Attacked Image Number")
        # Output the adv_image
        # Run the BIM attack
        adv_image = bim_atk.basic_iterative_method(model_fn=model, x=tf.convert_to_tensor(np.expand_dims(test_image, axis=0)),  # Add batch dimension
            eps=0.4, eps_iter=0.08, nb_iter=20, norm=np.inf, clip_min=0.0, clip_max=1.0, y=tf.convert_to_tensor([test_label]),
            targeted=False, rand_init=False, rand_minmax=0.3, #Initialize
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
        # print(f"Original Label: {orig_label}, Adversarial Label: {adv_label}, Total Perturbation (L2 norm): {tot_pert}")
        # print(f"Total Iterations: {10}")  # The number of iterations is specified by `nb_iter`
        atk_numbers = atk_numbers + 1
        print(atk_numbers)
        if adv_label != orig_label:
            eval_all.append(adv_image)
            labels.append(orig_label)
        if len(eval_all) % 1000 == 0:
            print("[INFO] Now Successful MIM Evaluation Num:", len(eval_all))
    
    print("[INFO] Success MIM Evalution Num:", len(eval_all))
    eval_all = tf.Variable(eval_all).numpy() # shape = (limit, 1, 28, 28)
    eval_all = eval_all.reshape(eval_all.shape[0], 28, 28, 1)
    np.savez(consts.BIM_EVAL_PATH.split('.')[0] + name + dataset + '.npz', eval=eval_all, labels=labels)

    return eval_all, labels
