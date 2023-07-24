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
        self.ytrain_encoded=to_categorical(self.ytrain,3)
        self.ytest_encoded=to_categorical(self.ytest,3)
        
        self.main_model=main_model
        self.filepath1='artifacts/main_model.h5'
        self.checkpoint1=ModelCheckpoint(self.filepath1, monitor='val_accuracy',save_best_only=True, 
                            mode='max', verbose=0)
        self.earlystopping1=EarlyStopping(monitor='val_loss',patience=10, verbose=0,mode='auto')
        self.callbacks=[self.checkpoint1,self.earlystopping1]
        self.epochs=100
        self.batch_size=32
        self.initial_epoch=11
        self.validation_data=(self.xtest,self.ytest_encoded)
        self.validation_split=0.33
    def initiate_model_training(self):
        try:
            

            trainig_set_classification_report,testing_set_classification_report,classified_samples,miss_classified_samples,confusion_matrix,model_hist=evaluate_model(
                                                                                    model_history=self.main_model,
                                                                                     xtrain=self.xtrain,
                                                                                     xtest=self.xtest,
                                                                                     ytrain=self.ytrain,
                                                                                     ytest=self.ytest,
                                                                                     ytrain_encoded=self.ytrain_encoded,
                                                                                     ytest_encoded=self.ytest_encoded,
                                                                                     epochs=self.epochs,
                                                                                     initial_epoch=self.initial_epoch,
                                                                                     batch_size=self.batch_size,
                                                                                     validation_data=self.validation_data,
                                                                                     callbacks=self.callbacks,
                                                                                     validation_split=self.validation_split)


            print('Training set classification report:\n{}'.format(trainig_set_classification_report))
            print('-'*100)
            print('Testining set classification report:\n{}'.format(trainig_set_classification_report))
            print('-'*100)
            print('Total mis-classified samples are: {}'.format(miss_classified_samples))
            print('Total classified samples are:{}'.format(classified_samples))
            print('-'*100)
            print('Confusion Matrix:\n{}'.format(confusion_matrix))
            fig, axs = plt.subplots(2, 1, figsize=(5, 5))

            # Plot Accuracy
            axs[0].plot(model_hist.history['accuracy'])
            axs[0].plot(model_hist.history['val_accuracy'])
            axs[0].set_title("Accuracy")
            axs[0].legend(['train', 'test'])

            # Plot Loss
            axs[1].plot(model_hist.history['loss'])
            axs[1].plot(model_hist.history['val_loss'])
            axs[1].set_title('Loss')
            axs[1].legend(['Train', 'Test'])

            # Display the subplots
            plt.tight_layout()
            plt.show()



        except Exception as e:
            raise custom_exception(e,sys)


