from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('results.html')

    else:
        data=CustomData(
            Gender=(request.form.get('Gender')),
            Age=float(request.form.get('Age')),
            Height=float(request.form.get('Height')),
            Weight=float(request.form.get('Weight')),
            family_history_with_overweight=(request.form.get('family_history_with_overweight')),
            FAVC=(request.form.get('FAVC')),
            FCVC=float(request.form.get('FCVC')),
            NCP=float(request.form.get('NCP')),
            CAEC=(request.form.get('CAEC')),
            SMOKE=(request.form.get('SMOKE')),
            CH2O=float(request.form.get('CH2O')),
            SCC=(request.form.get('SCC')),
            FAF=float(request.form.get('FAF')),
            TUE=float(request.form.get('TUE')),
            CALC=(request.form.get('CALC')),
            MTRANS=(request.form.get('MTRANS'))
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)
        
        categories={0:'INSUFFICIENT WEIGHT',1:'NORMAL WEIGHT',2:'OBESITY TYPE I',
                    3:'OBESITY TYPE II',4:'OBESITY TYPE III',5:'OVERWEIGHT LEVEL I',6:'OVERWEIGHT LEVEL II'}
        
        
        results = categories[result[0]]
        

        return render_template("results.html",results=results)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)
    
    
     
