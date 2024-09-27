from tensorflow import keras
from tensorflow.keras import layers, models
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet50V2


def VGG16(input_shape=(224, 224, 3), num_classes=7):
    input_tensor = layers.Input(shape=input_shape)

    # we define the block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # we define the block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # we define the block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # we define the block4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # we define the block5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # FLATTEN layer
    x = layers.Flatten(name='flatten')(x)

    # FC Layers
    # When dealing, the image size is too large, so we decrease the dense layer from 4096 to 1024
    x = layers.Dense(2048, activation='relu', name='fc1')(x)
    x = layers.Dense(2048, activation='relu', name='fc2')(x)

    # Output layer (adjust to the number of classes in PACS, domain generalization)
    x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create model
    model = models.Model(inputs=input_tensor, outputs=x)

    return model

# Example usage:
# vgg16_PACS = VGG16(input_shape=(224, 224, 3), num_classes=7)
# vgg16_PACS.summary()


def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    conv = layers.Conv2D(num_filters, 
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=keras.regularizers.l2(1e-4))
    
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    return x


def resnet_20(input_shape, depth, num_classes=7):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 40)')
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    
    inputs = layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block==0:
                strides = 2
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = layers.Activation('relu')(x)
        num_filters *= 2
        
    x = layers.AveragePooling2D(pool_size=8)(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, 
                    kernel_initializer='he_normal')(y)
    outputs = layers.Activation('softmax')(outputs)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model




def inceptionv3_block(x, filters):

    # The first inception block inception_a
    inception_a = layers.Conv2D(filters[0], (1,1), padding = 'same', activation = 'relu')(x)

    # The second inception block inception_b
    inception_b = layers.Conv2D(filters[1], (1,1), padding = 'same', activation = 'relu')(x)
    inception_b = layers.Conv2D(filters[2], (3,3), padding = 'same', activation = 'relu')(inception_b)

    # Two 3*3 convolution branch in the block
    inception_c = layers.Conv2D(filters[3], (1,1), padding = 'same', activation = 'relu')(x)
    inception_c = layers.Conv2D(filters[4], (3,3), padding = 'same', activation = 'relu')(inception_c)
    inception_c = layers.Conv2D(filters[4], (3,3), padding = 'same', activation = 'relu')(inception_c)

    # Maxpooling and convolution
    inception_d = layers.MaxPooling2D((3,3), strides = (1,1), padding = 'same')(x)
    inception_d = layers.Conv2D(filters[5], (1,1), padding = 'same', activation = 'relu')(inception_d)

    # Concatenate branch
    x = layers.concatenate([inception_a, inception_b, inception_c, inception_d], axis = -1)

    return x

def inception_v3(input_shape = (224,224,3), num_classes = 7):
    input_tensor = layers.Input(shape = input_shape)

    # Initial convolutional layers and pooling layers.
    x = layers.Conv2D(32, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(input_tensor)
    x = layers.Conv2D(32, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
    x = layers.Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = 'relu')(x)
    x = layers.MaxPooling2D((3,3), strides = (2,2), padding = 'same')(x)
    x = layers.Conv2D(80, (1,1), padding = 'same', activation = 'relu')(x)
    x = layers.Conv2D(192, (3,3), padding = 'same', activation = 'relu')(x)
    x = layers.MaxPooling2D((3,3), strides = (2,2), padding = 'same')(x)

    # Go through the inception block
    x = inceptionv3_block(x, [64, 48, 64, 64, 96, 32])
    x = inceptionv3_block(x, [64, 48, 64, 64, 96, 64])
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding = 'same')(x)

    x = inceptionv3_block(x, [128, 96, 128, 96, 128, 128])
    x = inceptionv3_block(x, [160, 128, 160, 128, 192, 128])
    x = inceptionv3_block(x, [192, 160, 192, 192, 256, 160])
    x = layers.MaxPooling2D((3,3), strides = (2,2), padding = 'same')(x)
    x = inceptionv3_block(x, [256, 160, 320, 160, 320, 160])

    # Add Global Average Pooling layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation = 'softmax')(x)

    return models.Model(input_tensor, x)


# model = inception_v3(input_shape = (224,224,3), num_classes = 7)
# model.summary()


# model = resnet_20(input_shape = (224,224,7), depth=20)
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
# model.summary()


# Begin DataLoader Session

# Then, we define the train_data generator
datagen = ImageDataGenerator(rescale=1./255)
# Then, we define the dataloader
def Dataloader(file_path, image_height, image_width, batch_size):
    # First, we define the data_generator using flow_from_directory 
    data_gen = datagen.flow_from_directory(file_path, target_size = (image_height, image_width), 
                                        batch_size = batch_size, class_mode = 'categorical', shuffle=False)
    x_all, y_all = [], []
    # i here stands for batch , for every batch
    for i in range(len(data_gen)):
        x_batch, y_batch = data_gen[i]
        x_all.append(x_batch)
        y_all.append(y_batch)
    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    return x_all, y_all

def Train_Dataloader(train_path, image_height, image_width, batch_size):
    x_train, y_train = [], []
    # Here, for three different domains in the train_path (photo, painting, cartoon)
    for domain in train_path:
        # Add data into train_data through Dataloader function
        x_data, y_data = Dataloader(domain, image_height, image_width, batch_size)
        # Append images to the dataset
        x_train.append(x_data)
        # Append label to the dataset
        y_train.append(y_data)
    
    # Combine them all into one numpy array
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return x_train, y_train

def Test_Dataloader(test_path, image_height, image_width, batch_size):
    return Dataloader(test_path, image_height, image_width, batch_size)


if __name__ == "__main__":

    # train_path = ['./PACS_data/sketch','./PACS_data/art_painting','./PACS_data/cartoon']
    # test_path = './PACS_data/photo'
    train_path = ['./PACS_data/sketch','./PACS_data/photo','./PACS_data/cartoon']
    test_path = './PACS_data/art_painting'
    image_height = 224
    image_width = 224
    batch_size = 32
    num_classes = 7



    # Load training data and testing data
    x_train, y_train = Train_Dataloader(train_path, image_height, image_width, batch_size)
    x_test, y_test = Test_Dataloader(test_path, image_height, image_width, batch_size)

    # Check the shape of all current input value
    print(f"x_train shape: {x_train.shape}, y_train shape:{y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # Start GPU initialization:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', choices=['vgg16', 'resnet-20','vgg16-test','resnet-20-test','pretrain_Resnet101', 'pretrain_Resnet50V2',"pretrain_Resnet50V2_test",
                                       'Inception_v3','Inception_v3-test','pretrain_Resnet50','pretrain_Resnet50_test'], help='models for training and testing')
    args = parser.parse_args()

    


    if args.m == "pretrain_Resnet50V2":
        model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224,224,3))
        x = model.output
        x = layers.GlobalAveragePooling2D()(x)
        # Add one more Dense layer
        x = layers.Dense(1024, activation='relu')(x)
        # Add one more Dropout layer to avoid overfitting
        x = layers.Dropout(0.5)(x)
        # Then, we add the final output layer
        output = layers.Dense(num_classes, activation='softmax')(x)
        Resnet50V2_PACS = models.Model(inputs = model.input, outputs = output)
        # Compile the corresponding model
        Resnet50V2_PACS.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                                loss = 'categorical_crossentropy',
                                metrics=['accuracy'])
        Resnet50V2_PACS.summary()
        # Add checkpoint_callback
        ckpoint_callback = ModelCheckpoint(f"./std_checkpoint/Pretrained_Res50V2_PACS.h5", monitor = 'val_accuracy',
                                            save_best_only=True, save_weights_only=False, mode ='max',verbose = 1)
        # Add early stopping callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        Resnet50V2_PACS.fit(x_train, y_train, epochs = 15, batch_size = 8, validation_data = (x_test, y_test), 
                callbacks = [ckpoint_callback,early_stopping_callback ])

    if args.m == "pretrain_Resnet101":
        model = ResNet101(weights='imagenet', include_top=False, input_shape=(224,224,3))
        x = model.output
        x = layers.GlobalAveragePooling2D()(x)
        # Add one more DENSE layer
        x = layers.Dense(1024, activation = 'relu')(x)
        # Add one more Dropout layer to avoid overfit
        x = layers.Dropout(0.5)(x)
        # Then, we add the final output layer
        output = layers.Dense(num_classes, activation='softmax')(x)
        Resnet101_PACS = models.Model(inputs = model.input, outputs = output)
        # Compile the corresponding model
        Resnet101_PACS.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        Resnet101_PACS.summary()
        # Add checkpoint_callback
        ckpoint_callback = ModelCheckpoint(f"./std_checkpoint/Pretrained_Res101_PACS.h5", monitor = 'val_accuracy',
                                            save_best_only=True, save_weights_only=False, mode ='max',verbose = 1)
        # Add early stopping callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        Resnet101_PACS.fit(x_train, y_train, epochs = 15, batch_size = 8, validation_data = (x_test, y_test), 
                callbacks = [ckpoint_callback,early_stopping_callback ])

    
    if args.m == "pretrain_Resnet50":
        model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
        x = model.output
        x = layers.GlobalAveragePooling2D()(x)
        # Add one more DENSE layer
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dense(1024, activation = 'relu')(x)
        # Add Dropout to avoid overfit
        x = layers.Dropout(0.5)(x) 
        # Then, we add final output luaer
        output = layers.Dense(num_classes, activation='softmax')(x)
        Resnet50_PACS = models.Model(inputs = model.input, outputs=output)

        # Compile model
        Resnet50_PACS.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
        # Add checkpoint_callback
        ckpoint_callback = ModelCheckpoint(f"./std_checkpoint/Pretrained_Res50_PACS.h5", monitor = 'val_accuracy',
                                            save_best_only=True, save_weights_only=False, mode ='max',verbose = 1)
        # Add early stopping callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        Resnet50_PACS.fit(x_train, y_train, epochs = 15, batch_size = 16, validation_data = (x_test, y_test), 
                callbacks = [ckpoint_callback,early_stopping_callback ])
        
    if args.m == "pretrain_Resnet50_test":
        Resnet50 = load_model("./std_checkpoint/Pretrained_Res50_PACS.h5")
        Resnet50.summary()
        test_loss, test_accuracy = Resnet50.evaluate(x_test, y_test, verbose=2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")

    if args.m == 'pretrain_Resnet50V2_test':
        Resnet50v2 = load_model("./std_checkpoint/Pretrained_Res50V2_PACS.h5")
        Resnet50v2.summary()
        test_loss, test_accuracy = Resnet50v2.evaluate(x_test, y_test, verbose =2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")

    if args.m == 'vgg16':
        model = VGG16(input_shape = (224,224,3), num_classes = 7)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])
        # Add checkpoint_callback
        ckpoint_callback = ModelCheckpoint(f"./std_checkpoint/VGG16_PACS.h5", monitor = 'val_accuracy',
                                            save_best_only=True, save_weights_only=False, mode ='max',verbose = 1)
        # Add early stopping callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        model.fit(x_train, y_train, epochs = 15, batch_size = 16, validation_data = (x_test, y_test), 
                callbacks = [ckpoint_callback,early_stopping_callback ])
        
    if args.m == 'vgg16-test':
        VGG16 = load_model("./std_checkpoint/VGG16_PACS.h5")
        VGG16.summary()
        test_loss, test_accuracy = VGG16.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")

    if args.m == 'resnet-20':
        model = resnet_20(input_shape = (224,224,3), depth=20)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), 
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])
        # Add checkpoint_callback
        ckpoint_callback = ModelCheckpoint(f"./std_checkpoint/Resnet20_PACS.h5", monitor = 'val_accuracy',
                                            save_best_only=True, save_weights_only=False, mode ='max',verbose = 1)
        # Add early stopping callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        # Start Training
        model.fit(x_train, y_train, epochs=20, batch_size = 16, validation_data = (x_test, y_test),
                  callbacks = [ckpoint_callback, early_stopping_callback])

    if args.m == 'resnet-20-test':
        resnet20 = load_model("./std_checkpoint/Resnet20_PACS.h5")
        resnet20.summary()
        test_loss, test_accuracy = resnet20.evaluate(x_test, y_test, verbose = 2)
        print(f"test loss equals to: {test_loss}")
        print(f"test accuracy equals to: {test_accuracy}")

    if args.m == 'Inception_v3':
        model = inception_v3(input_shape = (224,224,3), num_classes = 7)
        model.summary()
        model.compile(optimizer = Adam(learning_rate = 0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
        ckpoint_callback = ModelCheckpoint(f"./std_checkpoint/Inception_v3_PACS.h5", monitor = 'val_accuracy',
                                            save_best_only=True, save_weights_only=False, mode ='max',verbose = 1)
        # Add early stopping callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
        # Start Training
        model.fit(x_train, y_train, epochs=20, batch_size = 16, validation_data = (x_test, y_test),
                  callbacks = [ckpoint_callback, early_stopping_callback])