import numpy as np
np.random.seed(42)

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.layers import Dense, Conv2D, AveragePooling2D, MaxPooling2D, Activation, Add, Input, Flatten, BatchNormalization, Lambda
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.losses import sparse_categorical_crossentropy
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.applications import ResNet50
import os
import pickle
from subprocess import call

def identity_block(x_, depth, batch_norm=True):
    x = Conv2D(depth, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x_)
    x = Activation('elu')(x)
    if batch_norm is True:
        x = BatchNormalization(scale=True, center=True)(x)
    x = Conv2D(depth, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    x = Add()([x, x_])
    x = Activation('elu')(x)
    if batch_norm is True:
        x = BatchNormalization(scale=True, center=True)(x)
    return x

def create_model(num_conv=4, depth=16, batch_norm=True, resnet=False):
    img_in = Input((32,32,3), name='input')
    x = img_in
    
    if resnet is False:
        for i in range(num_conv):
            x = Conv2D(depth, (3, 3), padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
            x = Activation('elu')(x)
            if batch_norm is True:
                x = BatchNormalization(scale=True, center=True)(x)
    else:
        x = Conv2D(depth, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        x = Activation('elu')(x)
        if batch_norm is True:
            x = BatchNormalization(scale=True, center=True)(x)

        for i in range(num_conv/2):
            x = identity_block(x, depth, batch_norm)

    x = AveragePooling2D((32,32))(x)
    x = Flatten(name='flatten')(x)
    x = Dense(num_classes, name='classify')(x)
    x = Activation('softmax', name='activation_classify')(x)
    model = Model(img_in, x)
    return model

def train(model):
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
    checkpoints = [ReduceLROnPlateau(factor=.5, verbose=1),
                   #ModelCheckpoint(model_name, verbose=1, save_best_only=True),
                   EarlyStopping(patience=30, verbose=1)]
    hist = model.fit_generator(generator=datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs, verbose=1, validation_data=(x_test, y_test),
                        workers=4, callbacks=checkpoints)
    return hist


if __name__ == '__main__':
    batch_size = 100
    num_classes = 10
    epochs = 100
    data_augmentation = True
    mode = 'plain'
    model_name = mode + '.h5'
    regularizer = l2(1e-5)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255 - 0.5
    x_test = x_test.astype(np.float32) / 255 - 0.5

    for num_layers in range(11, 102, 10):
        model = create_model(11, depth=32, batch_norm=False, resnet=True)
        model.compile(optimizer=SGD(momentum=0.9, nesterov=True, clipnorm=1), loss=sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        model.summary()
        hist = train(model).history
