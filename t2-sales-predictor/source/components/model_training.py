import pandas as pd 
import numpy as np 
from source.exception import custom_exception
from source.loggers import logging
from dataclasses import dataclass
import os
import sys
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import accuracy_score
from source.utlis import evaluate_model
from source.utlis import save_obj
import matplotlib.pyplot as plt



class model_training:
    def __init__(self,main_model,train_array,test_array):
        self.xtrain=train_array[:,:-1]
        self.xtest=test_array[:,:-1]
        self.ytrain=train_array[:,-1]
        self.ytest=test_array[:,-1]
        
        self.main_model=main_model
        self.filepath1='artifacts/main_model.h5'
        self.checkpoint1=ModelCheckpoint(self.filepath1, monitor='val_mae',save_best_only=True, 
                            mode='auto', verbose=0)
        self.earlystopping1=EarlyStopping(monitor='val_loss',patience=10, verbose=0,mode='auto')
        self.callbacks=[self.checkpoint1,self.earlystopping1]
        self.epochs=100
        self.batch_size=16
        self.initial_epoch=11
        self.validation_data=(self.xtest,self.ytest)
        self.validation_split=0.22
    def initiate_model_training(self):
        try:
            

            model_started=evaluate_model(
                                                                                    model_history=self.main_model,
                                                                                     xtrain=self.xtrain,
                                                                                     xtest=self.xtest,
                                                                                     ytrain=self.ytrain,
                                                                                     ytest=self.ytest,
                                                                                    
                                                                                     epochs=self.epochs,
                                                                                     initial_epoch=self.initial_epoch,
                                                                                     batch_size=self.batch_size,
                                                                                     validation_data=self.validation_data,
                                                                                     callbacks=self.callbacks,
                                                                                     validation_split=self.validation_split)


            print('All get completed')

        except Exception as e:
            raise custom_exception(e,sys)


