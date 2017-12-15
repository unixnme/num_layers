import numpy as np
np.random.seed(42)

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.layers import Dense, Conv2D, AveragePooling2D, MaxPooling2D, Activation, Add, Input, Flatten, BatchNormalization, Lambda, Dropout
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.losses import sparse_categorical_crossentropy
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.applications import ResNet50
import os
import sys
import pickle
from subprocess import call
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

layer_idx = 1

def lr_decay(epoch):
    x = float(epoch) / epochs
    start = 1e-2
    end = 1e-4
    lr = end*x + start*(1-x)
    print 'lr = ' + str(lr)
    return lr

def identity_block(x_, depth, batch_norm=True, drop_rate=1.0, layers_per_block=2):
    global layer_idx
    x = x_

    for i in range(layers_per_block-1):
        x = Conv2D(depth, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer, bias_regularizer=regularizer,
                   name='conv_'+str(layer_idx))(x)
        x = Activation('elu', name='elu_'+str(layer_idx))(x)
        x = Dropout(drop_rate, name='dropout'+str(drop_rate)+'_'+str(layer_idx))(x)
        if batch_norm is True:
            x = BatchNormalization(scale=True, center=True, name='bn_'+str(layer_idx))(x)
        layer_idx += 1

    x = Conv2D(depth, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer,
               name='conv_'+str(layer_idx))(x)
    x = Add(name='add_'+str(layer_idx))([x, x_])
    x = Activation('elu', name='elu_'+str(layer_idx))(x)
    x = Dropout(drop_rate, name='dropout'+str(drop_rate)+'_'+str(layer_idx))(x)
    if batch_norm is True:
        x = BatchNormalization(scale=True, center=True, name='bn_'+str(layer_idx))(x)
    layer_idx += 1
    return x

def create_model(num_conv=4, depth=16, batch_norm=True, resnet=False, drop_rate=1.,
        layers_per_block=2):
    global layer_idx
    layer_idx = 1

    img_in = Input((32,32,3), name='input')
    x = img_in
    
    x = Conv2D(depth, (3, 3), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizer, bias_regularizer=regularizer,
               name='conv_'+str(layer_idx))(x)
    x = Activation('elu', name='elu_'+str(layer_idx))(x)
    x = Dropout(drop_rate, name='dropout'+str(drop_rate)+'_'+str(layer_idx))(x)
    if batch_norm is True:
        x = BatchNormalization(scale=True, center=True, name='bn_'+str(layer_idx))(x)
    layer_idx += 1

    if resnet is False:
        for i in range(num_conv):
            x = Conv2D(depth, (3, 3), padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=regularizer, bias_regularizer=regularizer,
                       name='conv_'+str(layer_idx))(x)
            x = Activation('elu', name='elu_'+str(layer_idx))(x)
            x = Dropout(drop_rate, name='dropout'+str(drop_rate)+'_'+str(layer_idx))(x)
            if batch_norm is True:
                x = BatchNormalization(scale=True, center=True, name='bn_'+str(layer_idx))(x)
            layer_idx += 1
    else:
        for i in range(num_conv/layers_per_block):
            x = identity_block(x, depth, batch_norm, drop_rate,
                    layers_per_block)

    x = AveragePooling2D((32,32), name='pooling')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(num_classes, name='dense')(x)
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
    checkpoints = [LearningRateScheduler(lr_decay)]
    hist = model.fit_generator(generator=datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs, verbose=1, validation_data=(x_test, y_test),
                        workers=4, callbacks=checkpoints)
    return hist


if __name__ == '__main__':
    batch_size = 100
    num_classes = 10
    epochs = 200
    depth = 32
    layers = 160
    data_augmentation = True
    regularizer = l2(1e-5)
    drop_rate = 0.1
    layers_per_block = [1, 5, 10, 20]
    name = 'resnet' + str(layers) + \
            '_lpb' + str(layers_per_block).replace(' ', '').replace('[', '').replace(']', '') + \
            '_depth' + str(depth) + \
            '_drop' + str(drop_rate)
    model_name = name + '.h5'

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255 - 0.5
    x_test = x_test.astype(np.float32) / 255 - 0.5

    with open('nohup.out.' + name, 'w') as output_file:
        sys.stdout =output_file

        hist = []
        if os.path.isfile(model_name):
            call(['rm', model_name])
        for lpb in layers_per_block:
            model = create_model(layers, depth=depth, batch_norm=True, resnet=True,
                    drop_rate=drop_rate, layers_per_block=lpb)
            plot_model(model, show_shapes=True)
            if os.path.isfile(model_name):
                model.load_weights(model_name, by_name=True)
            else:
                model.save(model_name)
            model.summary()
            model.compile(optimizer=SGD(momentum=0.9, nesterov=True), loss=sparse_categorical_crossentropy,
                          metrics=['accuracy'])
            hist.append(train(model).history)

        with open(name+'.p', 'wb') as f:
            pickle.dump(hist, f)

        #  "Accuracy"
        fig = plt.figure()
        axes = plt.gca()
        for i in range(len(layers_per_block)):
        plt.plot(hist[i]['acc'])
        plt.plot(hist[i]['val_acc'])
        axes.set_ylim([0, 1])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['resnet1 train', 'resnet1 val', 'resnet5 train', 'resnet5 val', 'resnet10 train', 'resnet10 val', 'resnet20 train', 'resnet20 val'], loc='lower right')
        fig.savefig(name + '_acc.png')


