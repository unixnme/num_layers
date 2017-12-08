from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.layers import Dense, Conv2D, AveragePooling2D, MaxPooling2D, Activation, Add, Input, Flatten
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.losses import sparse_categorical_crossentropy
import numpy as np
import os

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
model_name = 'keras_cifar10_trained_model.h5'
regularizer = l2(1e-5)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def identity_block(x_, depth):
    x = Conv2D(depth, (3,3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x_)
    x = Activation('relu')(x)
    x = Conv2D(depth, (3,3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    x = Activation('relu')(x)
    x = Conv2D(depth, (3,3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    x = Add()([x, x_])
    x = Activation('relu')(x)
    return x

def create_model():
    img_in = Input((32,32,3))
    x = img_in
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    x = Activation('relu')(x)
    for i in range(10):
        x = identity_block(x, 16)
    x = Flatten()(x)
    x = Dense(num_classes)(x)
    model = Model(img_in, x)
    model.summary()
    if os.path.isfile(model_name):
        model.load_weights(model_name, by_name=True)
    return model

def train():
    model = create_model()
    model.compile(optimizer=RMSprop(), loss=sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    steps_per_epoch = int(np.ceil(float(len(x_train)) / batch_size))
    model.fit_generator(generator=datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs, verbose=1, validation_data=(x_test, y_test),
                        workers=4)
    model.save(model_name)


if __name__ == '__main__':
    train()
