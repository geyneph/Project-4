#Importing dependecies 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

#Importing Data
df_mutation = pd.read_csv('METABRIC_RNA_Mutation.csv')
df_mutation = df_mutation.dropna()
#building dataFrame
noneGeneDF = df_mutation[[
 'age_at_diagnosis',
 'chemotherapy',
 'hormone_therapy',
 'mutation_count',
 'tumor_size',
 'tumor_stage',
 'overall_survival',
 ]]
noneGeneDF2 = df_mutation[[
'type_of_breast_surgery',
'cancer_type',
'cancer_type_detailed',
'cellularity']]
#Building dummies 
dummies = pd.get_dummies(noneGeneDF2,dtype=int)
#concatenating
result = pd.concat([noneGeneDF,dummies],axis=1)
result = result.reindex([
 'age_at_diagnosis',
 'chemotherapy',
 'hormone_therapy',
 'mutation_count',
 'tumor_size',
 'tumor_stage',
 'type_of_breast_surgery_BREAST CONSERVING',
 'type_of_breast_surgery_MASTECTOMY',
 'cancer_type_Breast Cancer',
 'cancer_type_detailed_Breast',
 'cancer_type_detailed_Breast Invasive Ductal Carcinoma',
 'cancer_type_detailed_Breast Invasive Lobular Carcinoma',
 'cancer_type_detailed_Breast Invasive Mixed Mucinous Carcinoma',
 'cancer_type_detailed_Breast Mixed Ductal and Lobular Carcinoma',
 'cellularity_High',
 'cellularity_Low',
 'cellularity_Moderate',
 'overall_survival'
 ],axis=1)
#Training the model
y = result['overall_survival']
X = result.copy()
X.drop('overall_survival',inplace=True,axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)
model = LogisticRegression(random_state=1)
# Fit the model using training data
model.fit(X_train, y_train)
#results
predictions = model.predict(X_test)
print(f'Model Score: {round(model.score(X_test, y_test),2)}')
print('-'*100)
print(classification_report(y_test, predictions))
print('-'*100)

#Flask coding 
app = Flask(__name__)
app.static_folder = 'static'
@app.route('/')
def index():
    return render_template('form1.html')



@app.route('/', methods = ['POST'])
def getvalue():
    name = request.form['name']
    age = request.form['age']
    chemotherapy = request.form['chemotherapy']
    hormone = request.form['Hormone']
    Nmutations = request.form['MutationsN']
    tumorsize = request.form['Tumorsize']
    BreastS = request.form['breastSurgery']
    CancerDetailed = request.form['cancerDetailed']
    Celullarity = request.form['Cellularity']
    tumorStage = request.form['Stage']
    #Processing userdata
    xusertest1 = {
        'age_at_diagnosis':float(age),
        'chemotherapy':float(chemotherapy),
        'hormone_therapy':float(hormone),
        'mutation_count':float(Nmutations),
        'tumor_size':float(tumorsize),
        'tumor_stage' :	float(tumorStage),
    }
    xusertest1= pd.DataFrame([xusertest1])
    print(xusertest1)
    xusertest2 = {
        'type_of_breast_surgery':str(BreastS),
        'cancer_type': 1 ,
        'cancer_type_detailed':str(CancerDetailed),
        'cellularity':str(Celullarity)
    }
    xusertest2 = pd.DataFrame([xusertest2])
    print(xusertest2)
    #Building dummies 
    dummiesuser = pd.get_dummies(xusertest2,dtype=int)
    #concatenating
    resultuser = pd.concat([xusertest1,dummiesuser],axis=1)
    resultuser = resultuser.reindex([
    'age_at_diagnosis',
    'chemotherapy',
    'hormone_therapy',
    'mutation_count',
    'tumor_size',
    'tumor_stage',
    'type_of_breast_surgery_BREAST CONSERVING',
    'type_of_breast_surgery_MASTECTOMY',
    'cancer_type_Breast Cancer',
    'cancer_type_detailed_Breast',
    'cancer_type_detailed_Breast Invasive Ductal Carcinoma',
    'cancer_type_detailed_Breast Invasive Lobular Carcinoma',
    'cancer_type_detailed_Breast Invasive Mixed Mucinous Carcinoma',
    'cancer_type_detailed_Breast Mixed Ductal and Lobular Carcinoma',
    'cellularity_High',
    'cellularity_Low',
    'cellularity_Moderate',
    ],axis=1)
    print(resultuser)
    resultuser = resultuser.fillna(0)
    #evaluating model with user data
    predictionUser = model.predict(resultuser)
    probaUser = model.predict_proba(resultuser)
    probaUser = probaUser.tolist()
    probaUser = round(probaUser[0][1]*100,2)

    if probaUser>50:
        return render_template(
            'result.html', 
            n=name,
            age = age,
            chemotherapy = chemotherapy,
            hormone = hormone,
            Nmutations = Nmutations,
            tumorsize = tumorsize,
            BreastS = BreastS,
            CancerDetailed = CancerDetailed,
            Celullarity = Celullarity,
            tumorStage = tumorStage,
            predictionUser = predictionUser,
            probaUser = probaUser
                            )
    elif probaUser<50:
        return render_template(
            'resultgood.html', 
            n=name,
            age = age,
            chemotherapy = chemotherapy,
            hormone = hormone,
            Nmutations = Nmutations,
            tumorsize = tumorsize,
            BreastS = BreastS,
            CancerDetailed = CancerDetailed,
            Celullarity = Celullarity,
            tumorStage = tumorStage,
            predictionUser = predictionUser,
            probaUser = probaUser
                            )

if __name__ == '__main__':
    app.run(debug=True)
