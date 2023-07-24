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
                                    ytrain_encoded,
                                        ytest_encoded,
                                            epochs,
                                                initial_epoch,
                                                    batch_size,
                                                        validation_data,
                                                            callbacks,
                                                                validation_split):
    try:
        final_model=model_history
        final_model_history=final_model.fit(xtrain,
                                            ytrain_encoded,
                                            epochs=epochs,
                                            initial_epoch=initial_epoch,
                                            batch_size=batch_size,
                                            validation_data=validation_data,
                                            callbacks=callbacks,
                                            validation_split=validation_split
                                            )
        ypred_test=np.argmax(final_model.predict(xtest),axis=-1)
        ypred_train=np.argmax(final_model.predict(xtrain),axis=-1)
        missclassified_samples=np.sum(ytest != ypred_test)
        classified_samples=np.sum(ytest== ypred_test)
        training_set_classification_report=classification_report(ytrain,ypred_train)
        testing_set_classification_report=classification_report(ytrain,ypred_train)
        cm_matrix=confusion_matrix(ytest,ypred_test)
        return (training_set_classification_report,
                testing_set_classification_report,
                classified_samples,
                missclassified_samples,
                cm_matrix,
                final_model_history)

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
                         ytrain_encoded,
                         ytest_encoded):
    try:
        tuner=RandomSearch(
                            hypermodel=build_model,
                            objective=objective,
                            max_trials=max_trails,
                            executions_per_trial=execution_per_trail,
                            directory=directory,
                            project_name=project_name)
        tuner.search(xtrain,ytrain_encoded,
             epochs=5,
             validation_data=(xtest,ytest_encoded))
        main_model=tuner.get_best_models(num_models=1)[0]

        return main_model
        
        
    except Exception as e:
        raise custom_exception(e)