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
import numpy as np
import os
import pickle
from subprocess import call

def identity_block(x_, depth, last_layer=True, batch_norm=True):
    x = Conv2D(depth, (3,3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x_)
    x = Activation('relu')(x)
    if batch_norm is True:
        x = BatchNormalization(center=False, scale=False)(x)

    x = Conv2D(depth, (3,3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    if last_layer is True:
        #x = Lambda(lambda x: -x)(x)
        x = Add()([x, x_])
        x = Activation('relu')(x)
        if batch_norm is True:
            x = BatchNormalization(center=False, scale=False)(x)
    return x

def recursive_identity_block(x_, depth, batch_norm=True):
    x = identity_block(x_, depth, batch_norm=batch_norm)
    x = identity_block(x, depth, batch_norm=batch_norm)
    x1 = identity_block(x, depth, last_layer=False, batch_norm=batch_norm)
    x = Add()([x_, x, x1])
    x = Activation('relu')(x)
    if batch_norm is True:
        x = BatchNormalization(center=False, scale=False)(x)
    return x

def create_recursive_residual_model(num_blocks=4, batch_norm=True):
    img_in = Input((32,32,3))
    x = img_in
    
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    x = Activation('relu')(x)

    for i in range(num_blocks):
        x = recursive_identity_block(x, depth=16, batch_norm=batch_norm)

    x = Flatten()(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    model = Model(img_in, x)
    '''
    if os.path.isfile(model_name):
        model.load_weights(model_name, by_name=True)
    '''
    plot_model(model, show_shapes=True)
    return model

def create_residual_model(num_blocks=4, batch_norm=True):
    img_in = Input((32,32,3))
    x = img_in
    
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    x = Activation('relu')(x)

    for i in range(num_blocks):
        x = identity_block(x, depth=16, batch_norm=batch_norm)

    x = Flatten()(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    model = Model(img_in, x)
    '''
    if os.path.isfile(model_name):
        model.load_weights(model_name, by_name=True)
    '''
    plot_model(model, show_shapes=True)
    return model

def create_plain_model(num_conv=4, batch_norm=True):
    img_in = Input((32,32,3))
    x = img_in
    
    for i in range(num_conv):
        x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
        x = Activation('relu')(x)
        if batch_norm is True:
            x = BatchNormalization(scale=True, center=True)(x)

    x = Flatten()(x)
    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    model = Model(img_in, x)
    '''
    if os.path.isfile(model_name):
        model.load_weights(model_name, by_name=True)
    '''
    plot_model(model, show_shapes=True)
    return model

def train(model):
    model.compile(optimizer=SGD(momentum=0.9, nesterov=True, clipnorm=1), loss=sparse_categorical_crossentropy,
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
    epochs = 1000
    data_augmentation = True
    model_name = 'num_conv.h5'
    regularizer = l2(1e-5)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255 - 0.5
    x_test = x_test.astype(np.float32) / 255 - 0.5

    history = []
    if os.path.isfile(model_name):
        print '\ndeleting weights\n'
        call(['rm', model_name])
    for num_conv in range(10, 101, 10):
        print '\n# of conv = ', str(num_conv), '\n'
        model = create_plain_model(num_conv=num_conv, batch_norm=False)
        if os.path.isfile(model_name):
            print '\nloading weights\n'
            model.load_weights(model_name, by_name=True)
        hist = train(model).history
        hist['num_conv'] = num_conv
        history.append(hist)
        model.save(model_name)
    with open('num_conv.p', 'wb') as f:
        pickle.dump(history, f)

    

