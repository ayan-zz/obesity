import os,sys
import pandas as pd
import numpy as np
from src.logging import logging
from src.exception import CustomException
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from src.utils import save_object,evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    logging.info('initiliazing model trainer path and pikel file')
    model_path=os.path.join('artifacts','model.pkl')
class model_trainer:
    def __init__(self):
        self.model_trainer_obj=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('initializing dependent and independent features for model training')
            X_train,X_test,y_train,y_test=(train_array[:,:-1],test_array[:,:-1],
                                           train_array[:,-1],test_array[:,-1])
            logging.info('providing models and hyperparameters')
            models={'LogisticRegression': LogisticRegression(),
                    'RandomForest':RandomForestClassifier(),
                    'GradientBoosting':GradientBoostingClassifier()}
            params={'LogisticRegression':{'max_iter':[1000],'penalty':['l2']},
                     'RandomForest':{'n_estimators': [100,500],'max_depth':[13],'min_samples_split':[5],'random_state':[1]},
                     'GradientBoosting':{'n_estimators': [100,500],'max_depth':[13],'min_samples_split' :[5],'random_state':[1]}}
            le=LabelEncoder()
            y_train=le.fit_transform(y_train)
            y_test=le.transform(y_test)

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                             X_test=X_test,y_test=y_test,models=models,params=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            
            save_object(
                file_path=self.model_trainer_obj.model_path,
                obj=best_model)
            
            predicted=best_model.predict(X_test)
            predicted=le.inverse_transform(predicted)
            print(f'The Best score of models:{best_model_name} : {best_model_score}')
            score= accuracy_score(y_test,predicted)
            logging.info('saving the best model as the model for prediction pipeline')
            return model_report
        
        except Exception as e:
            raise CustomException(e,sys)
        


