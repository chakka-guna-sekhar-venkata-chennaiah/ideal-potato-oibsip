import sys
import os
import pandas as pd
from source.loggers import logging
from source.exception import custom_exception
from source.utlis import load_object
from tensorflow.keras.models import load_model
import numpy as np

class prediction_pipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model=load_model('artifacts/main_model.h5')
            preprocessor_path='artifacts/preprocessor.pkl'
            
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaling=preprocessor.transform(features)
            prediction=np.argmax(model.predict(data_scaling),axis=-1)
            return prediction
            
        except Exception as e:
            raise custom_exception(e,sys)


    
class custom_data:
    def __init__(self,
        sepal_length:float,
        sepal_width:float,
        petal_length:float,
        petal_width:float,
        ):

        self.sepal_length=sepal_length
        self.sepal_width=sepal_width
        self.petal_length=petal_length
        self.petal_width=petal_width

    def get_data_as_a_dataframe(self):
        try:
            custom_data_input_dict={
            'sepal_length':[self.sepal_length],
            'sepal_width':[self.sepal_width],
            'petal_length':[self.petal_length],
            'petal_width':[self.petal_width]




            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise custom_exception(e,sys)
