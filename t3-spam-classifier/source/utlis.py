import numpy as np
import pandas as pd
import dill
import os
import sys
from source.exception import custom_exception
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import GridSearchCV
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

def evaluate_model(xtrain,xtest,ytrain,ytest,model,params):
    try:
        gs=GridSearchCV(model,params,cv=5,scoring='accuracy',n_jobs=-1,verbose=3)
        gs.fit(xtrain,ytrain)
        model.set_params(**gs.best_params_)
        model.fit(xtrain,ytrain)
        ytrain_pred=model.predict(xtrain)
        ytest_pred=model.predict(xtest)
        train_accuracy=accuracy_score(ytrain,ytrain_pred)
        test_accuracy=accuracy_score(ytest,ytest_pred)
        train_r2_score=r2_score(ytrain,ytrain_pred)
        test_r2_score=r2_score(ytest,ytest_pred)

        return (train_accuracy,test_accuracy,train_r2_score,test_r2_score)

    except Exception as e:
        raise custom_exception(e,sys)