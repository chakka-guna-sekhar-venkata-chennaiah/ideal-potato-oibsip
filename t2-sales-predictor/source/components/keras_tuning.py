import pandas as pd 
import numpy as np 
import random
from source.exception import custom_exception
from source.loggers import logging
from dataclasses import dataclass
from tensorflow.keras.utils import to_categorical
from keras_tuner.tuners import RandomSearch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU,LeakyReLU,ELU,PReLU,Softmax
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from source.utlis import perform_keras_tuning
import os
import sys



def build_model(hp):
    model=Sequential()
    units=hp.Int('units',min_value=9,max_value=729,step=9)
    bias_initiliazer=he_normal(seed=None)
    model.add(Dense(units=hp.Int('initial neurons for first layer',min_value=18,max_value=729,step=9,default=18),
                    kernel_initializer='he_normal',
                    use_bias=True,
                    bias_initializer=bias_initiliazer,
                    activation='relu',
                    input_dim=3))
    
    
    
    for i in range(hp.Int('num_layers',min_value=2,max_value=7,step=1)):
        
        model.add(Dense(units=units,
                            kernel_initializer='he_normal',
                            use_bias=True,
                            bias_initializer=bias_initiliazer,
                            activation='relu'))
           
    model.add(Dense(1,activation='relu'))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.Huber(),
                  metrics=['mae']
                  )
    return model

class keras_tuning:
    def __init__(self,train_array,test_array):
        
        self.xtrain=train_array[:,:-1]
        self.xtest=test_array[:,:-1]
        self.ytrain=train_array[:,-1]
        self.ytest=test_array[:,-1]
        
        self.hypermodel=build_model
        self.objective='val_mae'
        self.max_trails=5
        self.execution_per_trail=5
        self.directory='artifacts'
        self.project_name='task1'
    def start_keras_training(self):
        try:
            
            main_model=perform_keras_tuning(build_model=self.hypermodel,
                                            objective=self.objective,
                                            max_trails=self.max_trails,
                                            execution_per_trail=self.execution_per_trail,
                                            directory=self.directory,
                                            project_name=self.project_name,
                                            xtrain=self.xtrain,
                                            xtest=self.xtest,
                                            ytrain=self.ytrain,
                                            ytest=self.ytest)
            
            return main_model
        except Exception as e:
            raise custom_exception(e)
    

    
   

