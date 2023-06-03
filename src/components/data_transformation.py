import sys,os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.utils import save_object
from src.logging import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    logging.info('preparing a preprocessor pikel file')
    preprocessor_obj_path=os.path.join('artifacts','preprocessor.pkl')

class data_transformation:
    def __init__(self):
        self.data_transformation_obj=DataTransformationConfig()
    
    def get_transformation_obj(self):
        try:
            logging.info('ínitializing and creatng pipeline and preprocessor')
            num_col=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
            cat_col=['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
            
            num_pipeline=Pipeline(steps=(['imputer',SimpleImputer(strategy='median')],
                            ['scaler',StandardScaler()]))
            cat_pipeline=Pipeline(steps=(['imputer',SimpleImputer(strategy='most_frequent')],
                            ['encoder',OneHotEncoder()]))
            preprocessor=ColumnTransformer([('numerical_pipeline',num_pipeline,num_col),
                              ('categorical_pipeline',cat_pipeline,cat_col)])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_trasformation(self,train_path,test_path):
        try:
            logging.info('initializing data transformation')
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)
            le=LabelEncoder()
            target_col=['NObeyesdad']
            df_train['NObeyesdad']=le.fit_transform(df_train['NObeyesdad'])
            df_test['NObeyesdad']=le.transform(df_test['NObeyesdad'])
            target_feature_train_df=df_train[target_col]
            target_feature_test_df=df_test[target_col]

            logging.info('initializing preprocessor object')
            preprocessor_obj=self.get_transformation_obj()

            input_feature_train_df=df_train.drop(columns=['NObeyesdad'])
            input_feature_test_df=df_test.drop(columns=['NObeyesdad'])
           
            logging.info('Appliying preprocessor object on train and test file')
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,target_feature_train_df]
            test_arr=np.c_[input_feature_test_arr,target_feature_test_df]

            save_object(
                file_path=self.data_transformation_obj.preprocessor_obj_path,
                obj=preprocessor_obj
            )
            logging.info('Applied preprocessor obj formed and saved')
            return (train_arr,test_arr,self.data_transformation_obj.preprocessor_obj_path)

        except Exception as e:
            raise CustomException(e,sys)



