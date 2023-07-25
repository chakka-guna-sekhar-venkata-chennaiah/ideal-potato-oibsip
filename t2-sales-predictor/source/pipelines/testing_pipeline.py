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
            prediction=model.predict(data_scaling)
            prediction=prediction.flatten()
            return prediction
            
        except Exception as e:
            raise custom_exception(e,sys)


    
class custom_data:
    def __init__(self,
        tv:float,
        radio:float,
        newspaper:float
        ):

        self.tv=tv
        self.radio=radio
        self.newspaper=newspaper
        

    def get_data_as_a_dataframe(self):
        try:
            custom_data_input_dict={
            'tv':[self.tv],
            'radio':[self.radio],
            'newspaper':[self.newspaper]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise custom_exception(e,sys)
