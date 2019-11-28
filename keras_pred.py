from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GlobalAveragePooling2D, Activation, Average
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd 
import tensorflow as tf
import os
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import cifar10
from keras.engine import training
from keras.losses import categorical_crossentropy
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
from keras.models import load_model
import glob
import csv


train_image = np.load("Image_Processing/train_processed.npy")
train_label = pd.read_csv("data/train_labels.csv")
train_label = train_label.drop("Id",axis=1)

X_train = train_image[:32000]
y_train = train_label[:32000]
y_train = y_train.as_matrix()


X_validation = train_image[32000:36000]
y_validation = train_label[32000:36000]
y_validation = y_validation.as_matrix()

X_test = train_image[36000:]
y_test = train_label[36000:]
y_test = y_test.as_matrix()
#reshape 
img_rows , img_cols = 64, 64
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    input_shape = Input(shape=input_shape)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_validation = X_validation.reshape(X_validation.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    input_shape = Input(shape=input_shape)
#more reshaping
X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_test = X_test.astype("float32")

X_train /= 255
X_validation /= 255
X_test /= 255

num_category = 10
y_train = keras.utils.to_categorical(y_train, num_category)
y_validation = keras.utils.to_categorical(y_validation, num_category)


#################################### model 1 ###################################################################
def conv_1(input_shape: Tensor):
   
    ##conv 1
    x = Conv2D(32, (5, 5), activation='relu')(input_shape)
    x = MaxPooling2D((2,2))(x)

    ## conv2
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    #conv3
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    #conv4
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    
   
    x = Dropout(0.5)(x)
    x = Flatten()(x)           #flatten before dense layer , adding a channel size (batch,1)
    x = Dense(128,activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(input_shape, x,name= "conv_1")
    
    return model

model1 = conv_1(input_shape)

#################################### model 2 ###################################################################
def conv_2(input_shape: Tensor):
    
    x = Conv2D(32, (5, 5), activation='relu',padding='valid')(input_shape)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block2
    x = Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block3
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(32,  activation='relu')(x)
    
    x = Dense(10, activation='softmax')(x)
    
   
    
    model = Model(input_shape, x,name= "conv_2")
    
    return model

model2 = conv_2(input_shape)

################################### model 3 ################################################################
def conv_3(input_shape: Tensor): 
    
    x = Conv2D(32, (3,3), activation='relu',padding='valid')(input_shape)
    x = Conv2D(32, (3,3), activation='relu')(x)
    x = Conv2D(32, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block2
    x = Conv2D(64, (3,3), activation='relu',padding='valid')(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.5)(x)
    
    #mlpconv block3
    x = Conv2D(128, (3,3), activation='relu',padding='valid')(x)
    x = Conv2D(32, (3,3), activation='relu')(x)
    x = Conv2D(10, (3,3))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Activation(activation='softmax')(x)
    
    model = Model(input_shape, x,name = "conv_3")
    
    return model

model3 = conv_3(input_shape)

################################### model 5 ################################################################

def conv_5(input_shape:Tensor):
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(input_shape)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2,2),strides=(2, 2),)(x)
  
    
    #mlpconv block2
    x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2,2),strides=(2, 2),)(x)

    
    #mlpconv block3
    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D((2,2),strides=(2, 2),)(x)
    
    x = Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu',padding='same')(x)
    x = MaxPooling2D((2,2),strides=(2, 2),)(x)
    
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128,  activation='relu')(x)
    x = Dense(128,  activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    model = Model(input_shape, x,name= "conv_5")
    
    return model

model5 = conv_5(input_shape)




num_epoch = 10
batch_size = 64




def compile_model(model,num_epoch):
    

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    #model training
    model_log = model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=num_epoch,
                verbose=1,

                validation_data=(X_validation, y_validation))
    model_save = model.save("weights/"+model.name +".h5")

    
    return model_log,model_save


_, model1_weight_file = compile_model(model1, num_epoch)       #skip model_log
_, model2_weight_file = compile_model(model2, num_epoch)
_, model3_weight_file = compile_model(model3, num_epoch)     
_, model5_weight_file = compile_model(model5, num_epoch)

model2 = load_model("weights/"+"conv_2.h5")

model3 = load_model("weights/"+"conv_3.h5")

model1 = load_model("weights/"+"conv_1.h5") #90

model5 = load_model("weights/"+"conv_5.h5") #0.913

models = [model1,model2,model3,model5]

def ensemble(models,model_input: Tensor):
    outputs = [model(model_input) for model in models]
    y = Average()(outputs)  #compute average of all models' output
    
    model = Model(model_input, y, name='ensemble')
    
    return model
    
ensemble_model = ensemble(models, input_shape)


### check accuracy for the ensemble model using a sub set  test set from training set ##################
def valid_ensem(model):
    pred = model.predict(X_test)
    pred = np.argmax(pred,axis =1) #(4000,)
    pred = np.expand_dims(pred, axis=1)  #(4000,1)
    acc = np.sum(pred==y_test) / y_test.shape[0]  
    
    return acc


#valid_ensem(model2)
#valid_ensem(model3)
valid_ensem(ensemble_model)

##########model1 + model2 + model 5 = 0.9205



