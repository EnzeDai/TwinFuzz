import argparse
import tensorflow as tf
from tensorflow import keras
from models import LeNet5
from models import LeNet5_logits
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Reshape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# Normalization
x_train, x_test = x_train / 255.0, x_test / 255.0
# One-Hot Label
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

def preprocess_data():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Add the channel dimension to the images
    x_train = np.expand_dims(x_train, axis=-1)  # Shape: (num_samples, 28, 28, 1)
    x_test = np.expand_dims(x_test, axis=-1)    # Shape: (num_samples, 28, 28, 1)

    # Normalize the images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

def train_model(model, x_train, y_train, x_test, y_test, from_logits=False):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=64)
    model.evaluate(x_test, y_test)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices=['lenet5', 'lenet5_logits', 'lenet5_normal', 'resnet-20', 'googlenet', 'inception-v3', 'inception-resnet-v2', 'vgg16'], help='models for training')
    args = parser.parse_args()

    if args.m == 'lenet5':
        leNet5 = LeNet5()
        leNet5.compile(loss=tf.keras.losses.CategoricalCrossentropy, optimizer='adam', metrics=['accuracy'])
        leNet5.fit(x_train, y_train, epochs=10, batch_size=64)
        leNet5.evaluate(x_test, y_test)
        leNet5.save('./LeNet5_MNIST.h5')
    
    if args.m == 'lenet5_normal':
        # Preprocess the data
        x_train, y_train, x_test, y_test = preprocess_data()

        leNet5 = LeNet5()
        leNet5 = train_model(leNet5, x_train, y_train, x_test, y_test, from_logits=False)
        leNet5.save('./LeNet5_MNIST_normal.h5')

    
    if args.m == 'lenet5_logits':
        leNet5_logits = LeNet5_logits()
        leNet5_logits = train_model(leNet5_logits, x_train, y_train, x_test, y_test, from_logits=True)
        leNet5_logits.save('./LeNet5_MNIST_logits.h5')