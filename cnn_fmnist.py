import numpy as np
import pickle
import keras
from keras import backend as K, regularizers, optimizers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
from pca import pca, concat_bias
from numpy.random import seed
from tensorflow import set_random_seed
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.utils import np_utils
import os
from os import path

n_classes = 10
summary_file = './summary'
seed(1)
set_random_seed(2)

def get_avg(histories, his_key):
   tmp = []
   for history in histories:
       tmp.append(history[his_key][np.argmin(history['val_loss'])])
   return np.mean(tmp)

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def train(dir, optimizer, norm, lr, data, epochs, iteration, batch_size):
    # the data, shuffled and split between train and test sets
    if data == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape input shape for convolutional NN, one channel since gray
    row, col = 28, 28
    input_shape = (row, col, 1)
    X_train = X_train.reshape(X_train.shape[0], row, col, 1)
    X_test = X_test.reshape(X_test.shape[0], row, col, 1)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)

    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')


    model = build_cnn_model(input_shape)
    init_weight = {}
    layers = [f'conv2d_{2*(iteration-1) + i}' for i in range(1,3)] + [f'dense_{2*(iteration-1) + i}' for i in range(1,3)]
    for layer in layers:
        kernel, bias = model.get_layer(name=layer).get_weights()
        init_weight[layer] = concat_bias(kernel, bias)

    model.summary()

    model = multi_gpu_model(model)
    filepath = "{dir}/{{epoch:01d}}.hdf5".format(dir=dir)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                 save_best_only=False, mode='max')
    if not os.path.exists(dir):
        os.mkdir(dir)
    callbacks_list = [checkpoint]

    model.compile(optimizer=getattr(optimizers,optimizer)(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    init_loss = model.evaluate(X_train, Y_train)
    history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=0, validation_data=(X_test, Y_test), callbacks=callbacks_list)

    train_acc = history.history['acc']
    test_acc = history.history['val_acc']
    with open(summary_file, 'a') as f:
        f.write("%s\n%s %s\n" % (dir, train_acc[-1], test_acc[-1]))

    return init_weight, init_loss, history


def calc(data, optimizer, norm, lr, epochs, iteration, batch_size):
    dir = f'{batch_size}-weights-{optimizer}-{lr}-{norm}'
    init_weight, init_loss, history = train(dir=dir, data=data, optimizer=optimizer,
                                            norm=norm, lr=lr, epochs=epochs, iteration=iteration,
                                            batch_size=batch_size)

    prefix = f'/model_weights/sequential_{iteration}/'
    weights = {}
    index = 0
    for layer in init_weight.keys():
        weights[layer] = [np.array(init_weight[layer])]
        group_path = prefix + layer
        coordinates = pca(dir, group_path, weights[layer], epochs)
        with open(f'{dir}/store-{index}.pkl', 'wb') as f:
            pickle.dump([coordinates, history.history, init_loss], f)
        index += 1