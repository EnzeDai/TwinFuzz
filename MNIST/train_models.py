import argparse
import tensorflow as tf
from tensorflow import keras
from models import LeNet5

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices=['lenet5', 'resnet-20', 'googlenet', 'inception-v3', 'inception-resnet-v2', 'vgg16'], help='models for training')
    args = parser.parse_args()

    if args.m == 'lenet5':
        leNet5 = LeNet5()
        leNet5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        leNet5.fit(x_train, y_train, epochs=10, batch_size=64)
        leNet5.evaluate(x_test, y_test)
        leNet5.save('./LeNet5_MNIST.h5')
