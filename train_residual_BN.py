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

layer_counter = 1

def identity_block(x_, batch_norm=True):
    global layer_counter
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer,
               name='conv2d_'+str(layer_counter))(x_)
    x = Activation('relu', name='relu_'+str(layer_counter))(x)
    if batch_norm is True:
        x = BatchNormalization(scale=True, center=True)(x)
    layer_counter += 1

    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer,
               name='conv2d_'+str(layer_counter))(x)
    x = Add(name='add_'+str(layer_counter))([x_, x])
    x = Activation('relu', name='relu_'+str(layer_counter))(x)
    if batch_norm is True:
        x = BatchNormalization(scale=True, center=True)(x)
    layer_counter += 1
    return x

def create_residual_model(num_blocks, batch_norm=True):
    global layer_counter
    layer_counter = 1
    img_in = Input((32,32,3), name='input')
    x = img_in
    
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer,
               name='conv2d_'+str(layer_counter))(x)
    x = Activation('relu', name='relu_'+str(layer_counter))(x)
    if batch_norm is True:
        x = BatchNormalization(scale=True, center=True)(x)
    layer_counter += 1

    for i in range(num_blocks):
        x = identity_block(x, batch_norm=batch_norm)

    x = Flatten(name='flatten')(x)
    x = Dense(num_classes, name='classify')(x)
    x = Activation('softmax', name='activation_classify')(x)
    model = Model(img_in, x)
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
    epochs = 100
    data_augmentation = True
    mode = 'residual_BN'
    model_name = mode + '.h5'
    regularizer = l2(1e-5)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255 - 0.5
    x_test = x_test.astype(np.float32) / 255 - 0.5

    history = []
    if os.path.isfile(model_name):
        print '\ndeleting weights\n'
        call(['rm', model_name])
    for layer in range(1, 102, 10):
        print '\n# of layers = ', str(layer), '\n'
        num_blocks = (layer-1) / 2
        model = create_residual_model(num_blocks, batch_norm=True)
        model.summary()
        if os.path.isfile(model_name):
            print '\nloading weights\n'
            model.load_weights(model_name, by_name=True)
        hist = train(model).history
        hist['layer'] = layer
        history.append(hist)
        model.save(model_name)
    with open(mode+'.p', 'wb') as f:
        pickle.dump(history, f)

    

