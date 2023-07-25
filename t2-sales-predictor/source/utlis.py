import numpy as np
import pandas as pd
import dill
import os
import sys
from source.exception import custom_exception
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import GridSearchCV
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
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as fobj:
            dill.dump(obj,fobj)
            


    except Exception as e:
        raise custom_exception(e,sys)

def load_object(file_path):

    try:
        with open(file_path, 'rb') as fobj:
            return dill.load(fobj)
            
    except Exception as e:
        raise custom_exception(e,sys)

def evaluate_model(model_history,
                    xtrain,
                        xtest,
                            ytrain,
                                ytest,
                                    
                                            epochs,
                                                initial_epoch,
                                                    batch_size,
                                                        validation_data,
                                                            callbacks,
                                                                validation_split):
    try:
        final_model=model_history
        final_model_history=final_model.fit(xtrain,
                                            ytrain,
                                            epochs=epochs,
                                            initial_epoch=initial_epoch,
                                            batch_size=batch_size,
                                            validation_data=validation_data,
                                            callbacks=callbacks,
                                            validation_split=validation_split
                                            )
        
    except Exception as e:
        raise custom_exception(e,sys)


def perform_keras_tuning(build_model,
                        objective,
                        max_trails,
                        execution_per_trail,
                        directory,
                        project_name,
                         xtrain,
                         xtest,
                         ytrain,
                         ytest):
    try:
        tuner=RandomSearch(
                            hypermodel=build_model,
                            objective=objective,
                            max_trials=max_trails,
                            executions_per_trial=execution_per_trail,
                            directory=directory,
                            project_name=project_name)
        tuner.search(xtrain,ytrain,
             epochs=5,
             validation_data=(xtest,ytest))
        main_model=tuner.get_best_models(num_models=1)[0]

        return main_model
        
        
    except Exception as e:
        raise custom_exception(e)