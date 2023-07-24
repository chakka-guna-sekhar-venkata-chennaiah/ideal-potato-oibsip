import pandas as pd 
import numpy as np 
import random
import tensorflow as tf
from source.exception import custom_exception
from source.loggers import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys
from source.components.transformation import transformation_config
from source.components.transformation import transformation

from source.components.model_training import model_training
from source.components.keras_tuning import keras_tuning

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

@dataclass
class dataloadingconfig:
    train_data_path=os.path.join('artifacts','train_data.csv')
    test_data_path=os.path.join('artifacts','test_data.csv')
    raw_data_path=os.path.join('artifacts','data.csv')

class dataloading:
    def __init__(self):
        self.data_loading=dataloadingconfig()
    
    def initiate_data_loading(self):
        logging.info('Data loading is started')
        try:
            df=pd.read_csv('notebook/IRIS_1.csv')
            df['species']=df['species'].replace({'Iris-setosa':0,
                                                'Iris-versicolor':1,
                                                'Iris-virginica':2})

            logging.info('Reading the dataset as a dataframe')

            os.makedirs(os.path.dirname(self.data_loading.train_data_path),exist_ok=True)

            df.to_csv(self.data_loading.raw_data_path,index=False,header=True)
            logging.info('Train test split started')
            train_data,test_data=train_test_split(df,test_size=0.2,random_state=0)

            train_data.to_csv(self.data_loading.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_loading.test_data_path,index=False,header=True)

            logging.info('Data loading is completed')

            return(
                self.data_loading.train_data_path,
                self.data_loading.test_data_path
            )

        except Exception as e:
         raise custom_exception(e,sys)

if __name__=='__main__':
    data_load=dataloading()
    train_path,test_path=data_load.initiate_data_loading()

    data_transformation=transformation()
    train_array,test_array,_=data_transformation.initiate_transformer_object(train_path,test_path)

    keras_tuner=keras_tuning(train_array,test_array)
    main_model=keras_tuner.start_keras_training()

    modeltrainer=model_training(main_model,train_array,test_array)
    modeltrainer.initiate_model_training()

