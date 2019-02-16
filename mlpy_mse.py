# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 00:56:38 2018

@author: Kel3vra
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:43:47 2018

@author: Kel3vra
"""

'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:13:41 2018

@author: Kel3vra
"""


#from numpy.random import seed
#seed(10)
#from tensorflow import set_random_seed
#set_random_seed(10)
# =============================================================================

# =============================================================================
# 
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# # =============================================================================
# =============================================================================

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras import losses
from sklearn.metrics import accuracy_score, confusion_matrix,mean_absolute_error, classification_report, mean_squared_error
import seaborn as sn
import pandas as pd
from keras import metrics
from numpy.random import seed
import matplotlib.image as mpimg

import scipy.spatial as ss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
#from sklearn.metrics import DistanceMetric


#import NNfunctions as nn
#seed(2)
#from tensorflow import set_random_seed
#set_random_seed(2)

classes = [0,1,2,3,4,5,6,7,8,9]
# =============================================================================
# Number of seeds
# =============================================================================


batch_size = 128
num_classes = 10
epochs = 20


#we see our data
data =mnist.load_data()
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#x_train,x_test = nn.permutated(x_train,x_test)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(history.history, file=open('History.txt', 'w'))

y_pred = model.predict(x_test)
t = categories*y_pred
last =t.sum(axis=1)

y_pred_train = model.predict(x_train)


categories = np.unique(y_test_class)

y_train_class = np.argmax(y_train, axis=1)
y_pred_train_class = np.argmax(y_pred_train, axis=1)
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
mean_absolute_error(last, y_test_class)
mean_squared_error(last, y_test_class)
results = np.mean(y_test_class)
lasss= np.mean(last)
llast = len(last)
baseline  = np.tile(results, llast)
base_mae = mean_absolute_error(last, baseline)
base_mse = mean_squared_error(last, baseline)

print(np.flatnonzero(y_test_class != y_pred_class),file=open('Flat.txt', 'w'))
error= np.flatnonzero(y_test_class != y_pred_class)
print(y_test_class[error],file=open('CorrectDigits.txt', 'w'))
print(y_pred_class[error],file=open('WrongPredictions.txt', 'w'))

print(classification_report(y_test_class, y_pred_class),file=open('Measures.txt', 'w'))




confusion_mlp_RMSprop=confusion_matrix(y_test_class, y_pred_class)
print(confusion_mlp_RMSprop)

#nn.plot_confusion_matrix(cm           = confusion_mlp_RMSprop,
#                      normalize    = True,
#                      target_names  = classes,
#                      title        = "Confusion Matrix, test_Normalized")
#


true_values= np.flatnonzero(y_test_class == y_pred_class)
sum(y_test_class[true_values] == y_pred_class[true_values])


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
 