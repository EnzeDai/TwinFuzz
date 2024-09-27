import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from models import Dataloader, Train_Dataloader, Test_Dataloader


# ==== Configuration ====
name = "Resnet20"
train_path = ['./PACS_data/sketch','./PACS_data/art_painting','./PACS_data/cartoon']
test_path = './PACS_data/photo'
image_height = 224
image_width = 224
batch_size = 32
num_classes = 7

def load_PACS():
    x_train, y_train = Train_Dataloader(train_path, image_height, image_width, batch_size)
    x_test, y_test = Test_Dataloader(test_path, image_height, image_width, batch_size)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_PACS()
print(x_train.shape)
print(y_train.shape)

image = x_test[600]
# plt.imshow(image)
# plt.title(f"Label:{y_test[600]}")
# plt.axis('off')
# plt.savefig("ori_image.png", bbox_inches = 'tight')


def pgd_attack(model, inputs, labels, step, ep=0.3, epochs=1, isRand=True, randRate=1):

    in_cp = inputs.copy()
    target = tf.constant(labels)

    if step == None:
        step = ep / 8

    if isRand:
        inputs = inputs + np.random.uniform(-ep * randRate, ep * randRate)
        inputs = np.clip(inputs, 0, 1)
    
    # Specify the datatype

    in_adv = tf.Variable(inputs, dtype = tf.float32)
    for i in range(epochs):
        with tf.GradientTape() as tape:
            loss = keras.losses.categorical_crossentropy(target, model(in_adv))
            grads = tape.gradient(loss, in_adv)
        
        in_adv.assign_add(step * tf.sign(grads))
        in_adv = tf.clip_by_value(in_adv, clip_value_min=in_cp-ep, clip_value_max=in_cp+ep)
        in_adv = tf.clip_by_value(in_adv, clip_value_min=0, clip_value_max=1)
        in_adv = tf.Variable(in_adv)

    idxs = np.where(np.argmax(model(in_adv), axis=1) != np.argmax(target, axis=1))[0]
    print("[INFO] Success PGD Attack Num:", len(idxs))

    in_adv, in_cp, target = in_adv.numpy()[idxs], in_cp[idxs], target.numpy()[idxs]
    in_adv, target = tf.Variable(in_adv), tf.constant(target)

    return in_adv.numpy(), target.numpy()

model = keras.models.load_model(f"./std_checkpoint/{name}_PACS.h5")
test_img = np.expand_dims(x_test[600], axis = 0)
label_image = np.expand_dims(y_test[600], axis = 0)
advs, labels = pgd_attack(model, test_img, label_image, step=None)


advs, test_img = advs.squeeze(), test_img.squeeze()
print(type(advs))
print(type(test_img))

if test_img.dtype == np.float32 or test_img.dtype == np.float64:
    original_image = np.clip(test_img, 0, 1)
    adv_image = np.clip(advs, 0, 1)
else:
    original_image = np.clip(test_img, 0, 255) / 255.0
    adv_image = np.clip(advs, 0, 255) / 255.0

perturbation = adv_image - original_image
perturbation = np.clip(perturbation, -1, 1)
print(adv_image.shape)
print(original_image.shape)
print(perturbation)
plt.imshow(perturbation, vmin=-1, vmax = 1)

# Visualization:
# adv_image = np.squeeze(advs, axis = 0)
# plt.imshow(adv_image)
# plt.title(f"Label:{labels}")
# plt.savefig("adv_image.png", bbox_inches = 'tight')

# Visualization of the perturbations

plt.savefig("perturbation.png", bbox_inches='tight')






