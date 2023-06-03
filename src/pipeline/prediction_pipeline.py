import sys,os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessr=load_object(file_path=preprocessor_path)
            data_scaled=preprocessr.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
              raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, 
                 Gender : str,
                 Age: float,
                 Height :float,
                 Weight: float,
                 family_history_with_overweight: str,
                 FAVC: str,
                 FCVC: float,
                 NCP: float,
                 CAEC: str,
                 SMOKE: str,
                 CH2O: float,
                 SCC: str,
                 FAF: float,
                 TUE: float,
                 CALC: str,
                 MTRANS: str):
        
        self.Gender = Gender
        self.Age = Age
        self.Height = Height
        self.Weight = Weight
        self.family_history_with_overweight=family_history_with_overweight
        self.FAVC=FAVC
        self.FCVC=FCVC
        self.NCP = NCP
        self.CAEC= CAEC
        self.SMOKE= SMOKE
        self.CH2O=CH2O
        self.SCC=SCC
        self.FAF=FAF
        self.TUE=TUE
        self.CALC=CALC
        self.MTRANS=MTRANS
                
    def get_data_as_dataframe(self):
          try:
                custom_data_input_dict={
                      "Gender":[self.Gender],
                      "Age":[self.Age],
                      "Height":[self.Height],
                      "Weight":[self.Weight],
                      "family_history_with_overweight":[ self.family_history_with_overweight],
                      "FAVC":[self.FAVC],
                      "FCVC":[self.FCVC],
                      "NCP":[self.NCP],
                      "CAEC":[self.CAEC],
                      "SMOKE":[self.SMOKE],
                      "CH2O":[self.CH2O],
                      "SCC":[self.SCC],
                      "FAF":[self.FAF],
                      "TUE":[self.TUE],
                      "CALC":[self.CALC],
                      "MTRANS":[self.MTRANS]        
                }
                
                return pd.DataFrame(custom_data_input_dict)
          
          except Exception as e:
                raise CustomException(e,sys)
          