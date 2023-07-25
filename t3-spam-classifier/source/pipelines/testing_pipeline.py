import sys
import os
import pandas as pd
from source.loggers import logging
from source.exception import custom_exception
from source.utlis import load_object
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string  
import numpy as np
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

class prediction_pipeline:
    def __init__(self):
        pass

    def predict(self,input_sms):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            transformed_sms = transform_text(input_sms)
            
            data={'transformed_text':[transformed_sms]}
            df=pd.DataFrame(data)
            vector_input=preprocessor.transform(df).toarray()
            prediction=model.predict(vector_input)[0]
            return prediction
           
            
        except Exception as e:
            raise custom_exception(e,sys)


    
