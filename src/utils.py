import pandas as pd
import numpy as np
import os,sys
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,f1_score
from src.logging import logging
from src.exception import CustomException

def save_object(file_path,obj):
    try:
        logging.info("file path initialization started")
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,X_test,y_train,y_test,models,params):
    try:
        report={}
        le=LabelEncoder()
        y_train=le.fit_transform(y_train)
        y_test=le.transform(y_test)
        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]
            
            gs=GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_test_pred=model.predict(X_test)
            acc_score=(accuracy_score(y_test,y_test_pred))*100
            
            report[list(models.keys())[i]]=acc_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)