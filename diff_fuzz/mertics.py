import numpy as np
import tensorflow as tf
from tensorflow import keras

# FOL and entropy metrics

# FOL for L_infinity Norm
def fol_Linf(model, x, xi, ep, y):
    x, label = tf.Variable(x), tf.constant(y)
    fols = []
    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(label, model(x))
        grads = tape.gradient(loss, x)
        norm = np.linalg.norm(grads.numpy().reshape(x.shape[0], -1), ord=1, axis=1)
        flat = grads.numpy().reshape(x.shape[0], -1)       
        diff = (x.numpy() - xi).reshape(x.shape[0], -1)
        for i in range(x.shape[0]):
            i_fol = -np.dot(flat[i], diff[i]) + ep * norm[i]
            fols.append(i_fol)
    
    return np.array(fols)


# FOL for L_2 Norm 
def fol_L2(model, x, y):
    x, label = tf.Variable(x), tf.constant(y)
    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(label, model(x))
        grads = tape.gradient(loss, x)
        grads_norm_L2 = np.linalg.norm(grads.numpy().reshape(x.shape[0], -1), ord=2, axis=1)

    return  grads_norm_L2

# Entropy for generalization enhance
def entropy(prob):
    '''
        prob - softmax probability
    '''
    ep = 1e-10 # avoid NaN value
    prob = tf.clip_by_value(prob, ep, 1.0)
    entro = prob * tf.math.log(prob)
    entro = -1.0 * entro
    entro = tf.reduce_sum(entro)

    return entro