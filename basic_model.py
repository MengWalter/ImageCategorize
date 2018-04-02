#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:30:03 2018

@author: williammeng
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import os
import glob as gb
import numpy as np
PATH = os.getcwd()

def getdata(path):
    data_path = gb.glob(path +"/*.npy") 
    for i in range(len(data_path)):
        if i == 0:
            x = np.load(data_path[i])
            y = np.zeros([1,len(x)])
        else:
            imdata = np.load(data_path[i])
            x = np.vstack((x,imdata))
            sy = np.ones([1,len(imdata)])*i
            y = np.hstack((y,sy))
    return x,y



def main():
    
    path_train = PATH + '/Training data'
    path_val = PATH + '/Validate data'

    tx,ty = getdata(path_train)
    vx,vy = getdata(path_val)
    
    x_train = (tx-127.5)/255.0 
    y_train = keras.utils.to_categorical(ty.transpose(), num_classes=18)
    #y_train = ty.transpose()
    x_val = (vx-127.5)/255.0 
    y_val = keras.utils.to_categorical(vy.transpose(), num_classes=18)
    #y_val = vy.transpose()
    
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation, :]
    y_train = y_train[permutation]
    
    
    permutation2 = np.random.permutation(x_val.shape[0])
    x_val = x_val[permutation2, :]
    y_val = y_val[permutation2]
    
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
   
    model.add(Dense(720, activation='relu', input_dim=2500))
    model.add(Dropout(0.4))
    model.add(Dense(18, activation='softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128,
              validation_data=(x_val, y_val))
        
    score = model.evaluate(x_val, y_val, batch_size=128)
    print('test score:',score)
    
    model.save_weights('First_try.h5')
    
    path_pre = 'predict.npy'
    pre_x = np.load(path_pre)
    pre_x = (pre_x -127.5)/255.0
    prevalue = model.predict(pre_x,batch_size=128)
    preclass = model.predict_classes(pre_x)
    preprob = model.predict_proba(pre_x,batch_size=128)

    np.save("prediction.npy", prevalue)
    np.save("predictclass.npy",preclass)
    np.save("predictprob.npy",preprob)
    print(prevalue.shape)
    print(preclass.shape)
    print(preprob.shape)


if __name__ == '__main__':
    main() 


