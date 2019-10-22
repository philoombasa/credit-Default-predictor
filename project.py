from flask import Flask,render_template,url_for,request
from flask_material import Material
#EDA Packages
import pandas as pd
import numpy as np
#ml packages
from sklearn.externals import joblib
app = Flask(__name__)
Material(app)


@app.route('/')
def  index():
    return render_template("index.html")
@app.route('/preview')
def  preview():
    df=pd.read_csv("data/loans.csv")
    df1=df.head(20)
    return render_template("preview.html",df_view=df1)
@app.route('/analyze',methods=["POST"])
def  analyze():
    if request.method=='POST':
        CRBScore=request.form["CRBScore"]
        Age=request.form["Age"]
        Amount=request.form["Amount"]
        MonthsSinceOpen=request.form["MonthsSinceOpen"]
        FinancialMeasure1=request.form["FinancialMeasure1"]
        FinancialMeasure2=request.form["FinancialMeasure2"]
        FinancialMeasure4=request.form["FinancialMeasure4"]
        CustomerType=request.form["CustomerType"]
        SOR=request.form["SOR"]
        model_choice=request.form["model_choice"]

        sample_data=[CRBScore,Age,Amount,MonthsSinceOpen,FinancialMeasure1,FinancialMeasure2,FinancialMeasure4,CustomerType,SOR]
        #unicode to float
        clean_data=[float(i)for i in sample_data]
         #reshape the data as a sample not individual features
        ex1=np.array(clean_data).reshape(1,-1)
         #reloading the model
        if model_choice=="LogisticRegression_model":
             logit_model=joblib.load('data/LogisticRegression_model.pkl')
             result_prediction=logit_model.predict(ex1)
        elif model_choice=="RandomForest_model":
             rfmodel=joblib.load('data/RandomForest_model.pkl')
             result_prediction=rfmodel.predict(ex1)
        elif model_choice=="DecisionTree_model":
             DecisionTree_model=joblib.load('data/DecisionTree_model.pkl')
             result_prediction=DecisionTree_model.predict(ex1)
        elif model_choice=="XGBoost_model":
             xgbmodel=joblib.load('data/xgb_model.pkl')
             result_prediction=xgbmodel.predict(ex1)
    return render_template("index.html",CRBScore=CRBScore,
    Age=Age,Amount=Amount,
    MonthsSinceOpen=MonthsSinceOpen,
    FinancialMeasure1=FinancialMeasure1,
    FinancialMeasure2=FinancialMeasure2,
    FinancialMeasure4=FinancialMeasure4,
    CustomerType=CustomerType,
    SOR=SOR,
    clean_data=clean_data,
    result_prediction=result_prediction,
    model_selected=model_choice)

if __name__ == '__main__':
    app.run(debug=True)
