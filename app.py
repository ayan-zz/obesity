from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
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
            Age=(request.form.get('Age')),
            Height=(request.form.get('Height')),
            Weight=(request.form.get('Weight')),
            family_history_with_overweight=(request.form.get('Profamily_history_with_overweightperty_Area')),
            FAVC=(request.form.get('FAVC')),
            FCVC=float(request.form.get('FCVC')),
            NCP=float(request.form.get('NCP')),
            CAEC=float(request.form.get('CAEC')),
            SMOKE=float(request.form.get('SMOKE')),
            CH20=float(request.form.get('CH20')),
            SCC=float(request.form.get('SCC')),
            FAF=float(request.form.get('FAF')),
            TUE=float(request.form.get('TUE')),
            CALC=float(request.form.get('CALC')),
            MTRANS=float(request.form.get('MTRANS'))
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)

        categories={0:'NOT APPROVED',1:'APPROVED'}
        
        results = result
    #categories[result[0]]
        

        return render_template("results.html",results=results)

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True,port=5000)
    
    
     
