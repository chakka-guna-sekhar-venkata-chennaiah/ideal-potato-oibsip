import pandas as pd 
import numpy as np 
from source.exception import custom_exception
from source.loggers import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
import sys
from source.components.transformation import transformation_config
from source.components.transformation import transformation
from source.components.model_training import model_training_config
from source.components.model_training import model_training
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string  
ps = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')

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

            df=pd.read_csv('notebook/spam.csv',encoding='latin1')
            df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
            df.rename(columns={'v1':'target','v2':'text'},inplace=True)
            df['target']=df['target'].replace({'spam':1,
                                                'ham':0
                                                })

            df.drop_duplicates(keep='first')
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

            df['transformed_text'] = df['text'].apply(transform_text)
            df.drop(columns=['text'],inplace=True,axis=1)

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

    modeltrainer=model_training()
    modeltrainer.initiate_model_training(train_array,test_array)

