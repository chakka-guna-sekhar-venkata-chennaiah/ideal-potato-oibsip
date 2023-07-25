import pandas as pd 
import numpy as np 
from source.exception import custom_exception
from source.loggers import logging
from dataclasses import dataclass
import os
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from source.utlis import evaluate_model
from source.utlis import save_obj

@dataclass
class model_training_config:
    trained_model_path=os.path.join('artifacts','model.pkl')

class model_training:
    def __init__(self):
        self.model_training_config=model_training_config()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting training and testing inputs')
            xtrain,xtest,ytrain,ytest=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1])
            model=MultinomialNB()
            
            lr_param={
                'alpha': [0.01, 0.1, 0.5, 1.0, 2.0] 
                    }
            params=[lr_param]


            train_accuracy,test_accuracy,train_r2_score,test_r2_score=evaluate_model(xtrain=xtrain,xtest=xtest,ytrain=ytrain,ytest=ytest,
                                                                        model=model,params=params)

            save_obj(
                file_path=self.model_training_config.trained_model_path,
                obj=model
            )

            print('Training accuracy is: ',train_accuracy)
            print('Testing accuracy is: ',test_accuracy)
            print('Train r2_score is: ',train_r2_score)
            print('Test r2_score is: ',test_r2_score)


        except Exception as e:
            raise custom_exception(e,sys)


