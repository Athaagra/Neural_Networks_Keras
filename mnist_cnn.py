# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:13:41 2018

@author: Kel3vra
"""

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import NNfunctions as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
batch_size = 128
num_classes = 10
epochs = 12

classes = [0,1,2,3,4,5,6,7,8,9]
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=False)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(history.history, file=open('History.txt', 'w'))

y_pred = model.predict(x_test)
y_pred_train = model.predict(x_train)




y_train_class = np.argmax(y_train, axis=1)
y_pred_train_class = np.argmax(y_pred_train, axis=1)
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)


print(classification_report(y_train_class, y_pred_train_class))
  
 
print(classification_report(y_test_class, y_pred_class),file=open('Measures.txt', 'w'))
confusion_mlp_RMS=confusion_matrix(y_train_class, y_pred_train_class)
print(confusion_mlp_RMS)
nn.plot_confusion_matrix_train(cm           = confusion_mlp_RMS,
                      normalize    = True,
                      target_names  = classes,
                      title        = "Confusion Matrix, train_Normalized")



confusion_mlp_RMSprop=confusion_matrix(y_test_class, y_pred_class)
print(confusion_mlp_RMSprop)

nn.plot_confusion_matrix(cm           = confusion_mlp_RMSprop,
                      normalize    = True,
                      target_names  = classes,
                      title        = "Confusion Matrix, test_Normalized")
# =============================================================================
# Confusion Matrix Heatmap
# =============================================================================
print(np.flatnonzero(y_test_class != y_pred_class),file=open('Flat.txt', 'w'))
error= np.flatnonzero(y_test_class != y_pred_class)
def images(error):
    prob_error = y_pred[error][y_pred_class[error]] - y_pred[error][y_test_class[error]]
    plt.imshow(x_test[error].reshape(28,28), cmap='gray', interpolation='none')
    plt.title( "Predicted {}, Truth: {}, Probability error:{}, index:{}".format(y_pred_class[error], y_test_class[error], "%.4f" % prob_error,error))
    plt.xticks([])
    plt.yticks([])  
    plt.show()
    return y_pred_class[error],y_test_class[error], prob_error, error


img = [images(i) for i in error]   
  
imgs = sorted(img, key=lambda item: item[2]) 
print(imgs,file=open('imgs.txt', 'w'))
print(len(imgs),file=open('lenimgs.txt', 'w'))