import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

# ==== Configuration ====
name = "LeNet5"


 # Fast Gradient Sign Method
def fgsm_attack(model, inputs, labels, ep=0.3, isRand=True, randRate=1):
    
    in_cp = inputs.copy() # shallow copy, reference only
    target = tf.constant(labels)
    
    if isRand:
        inputs = np.random.uniform(-ep * randRate, ep * randRate, inputs.shape)
        inputs = np.clip(inputs, 0, 1)

    inputs = tf.Variable(inputs)
    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(target, model(inputs))
        grads = tape.gradient(loss, inputs)
    
    in_adv = inputs + ep * tf.sign(grads)
    in_adv = tf.clip_by_value(in_adv, clip_value_min=in_cp-ep, clip_value_max=in_cp+ep)
    in_adv = tf.clip_by_value(in_adv, clip_value_min=0, clip_value_max=1)

    idxs = np.where(np.argmax(model(in_adv), axis=1) != np.argmax(labels, axis=1))[0]
    print("Success FGSM Attack Num:", len(idxs))

    in_adv, in_cp, target = in_adv.numpy()[idxs], in_cp[idxs], target.numpy()[idxs]
    in_adv, target = tf.Variable(in_adv), tf.constant(target)

    return in_adv.numpy(), target.numpy()


# Projected Gradient Descent
def pgd_attack(model, inputs, labels, step, ep=0.3, epochs=10, isRand=True, randRate=1):

    in_cp = inputs.copy()
    target = tf.constant(labels)

    if step == None:
        step = ep / 8

    if isRand:
        inputs = inputs + np.random.uniform(-ep * randRate, ep * randRate)
        inputs = np.clip(inputs, 0, 1)
  
    in_adv = tf.Variable(inputs)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(target, model(in_adv))
            grads = tape.gradient(loss, in_adv)
        
        in_adv.assign_add(step * tf.sign(grads))
        in_adv = tf.clip_by_value(in_adv, clip_value_min=in_cp-ep, clip_value_max=in_cp+ep)
        in_adv = tf.clip_by_value(in_adv, clip_value_min=0, clip_value_max=1)
        in_adv = tf.Variable(in_adv)

    idxs = np.where(np.argmax(model(in_adv), axis=1) != np.argmax(target, axis=1))[0]
    print("Success PGD Attack Num:", len(idxs))

    in_adv, in_cp, target = in_adv.numpy()[idxs], in_cp[idxs], target.numpy()[idxs]
    in_adv, target = tf.Variable(in_adv), tf.constant(target)

    return in_adv.numpy(), target.numpy()


# Format data structure
def data_format(x_train, x_test, y_train, y_test):

    # Reshape
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # One-Hot Label
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test


# Use dataset train data to generate adv samples.
def gen_adv_samples():

    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    model = keras.models.load_model(f"./{name}_MNIST.h5")

    x_train, x_test, y_train, y_test = data_format(x_train, x_test, y_train, y_test)

    advs, labels = fgsm_attack(model, x_train, y_train)
    np.savez('./FGSM_AdvTrain.npz', advs=advs, labels=labels)

    advs, labels = pgd_attack(model, x_train, y_train, step=None)
    np.savez('./PGD_AdvTrain.npz', advs=advs, labels=labels)

    return


# Standard adversarial training using PGD and FGSM
def adv_train():

    fgsm_advpath = "FGSM_AdvTrain.npz"
    pgd_advpath = "PGD_AdvTrain.npz"

    if os.path.exists(fgsm_advpath) and os.path.exists(pgd_advpath):
        print('[INFO]: Adv samples are generated.')
    else:
        gen_adv_samples()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test, y_train, y_test = data_format(x_train, x_test, y_train, y_test)

    with np.load(fgsm_advpath) as f:
        fgsm_train, fgsm_labels = f['advs'], f['labels']
    with np.load(pgd_advpath) as f:
        pgd_train, pgd_labels = f['advs'], f['labels']

    adv_train = np.concatenate((fgsm_train, pgd_train))
    adv_labels = np.concatenate((fgsm_labels, pgd_labels))

    # incremental adv train, adv_num / clean_num = 20%
    sampleNum = [600*i for i in [1, 2, 4, 8, 12, 16, 20]]

    for n in sampleNum:
        model_ckpoint = f"./checkpoint/{name}_MNIST_Adv_{n}.h5"
        ori_model = keras.models.load_model(f"./{name}_MNIST.h5")

        checkpoint = ModelCheckpoint(filepath=model_ckpoint, monitor='val_accuracy', verbose=0, save_best_only=True)
        callbacks = [checkpoint]

        indexes = np.random.choice(len(adv_train), n)
        selectAdv = adv_train[indexes]
        selectLabel = adv_labels[indexes]

        x_train_adv = np.concatenate((x_train, selectAdv), axis=0)
        y_train_adv = np.concatenate((y_train, selectLabel), axis=0)

        # Now retraining
        ori_model.fit(x_train_adv, y_train_adv, epochs=10, batch_size=64, verbose=0, callbacks=callbacks)

        best_resist_model = keras.models.load_model(model_ckpoint)

        _, acc_clean = best_resist_model.evaluate(x_test, y_test, verbose=0)
        print("After adv training, clean accuracy:", acc_clean)

    return

if __name__ == "__main__":
    gen_adv_samples()
    adv_train()
