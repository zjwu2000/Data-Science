# -*- coding: utf-8 -*-
"""
Created on Nov 24 07:10:06 2017

@author: Joe

revision history
v1:
Accuracy : 83.225%
Cross-Validation Score : 80.948%

v2: normalize
Accuracy : 82.736%
Cross-Validation Score : 80.784%

v3: added LoanAmountPerIncome
Accuracy : 83.225%
Cross-Validation Score : 79.805%

V4:
Accuracy : 84.853%
Cross-Validation Score : 82.251%
"""

import pandas as pd
import numpy as np
import sys, os


pd.set_option('display.max_rows', 1000)

df = pd.read_csv(r"C:\Data Science Projects\Load Prediction\loan_train_data.csv", sep=',', header=0, decimal='.')
df_test = pd.read_csv(r"C:\Data Science Projects\Load Prediction\loan_test_data.csv", sep=',', header=0, decimal='.')


print(df.head(10))

df.describe()

#Create a new function:
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print("(train)Missing values per column:")
print(df.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column
print("(test)Missing values per column:")
print(df_test.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

#Applying per row:
#print("\nMissing values per row:")
#print(df.apply(num_missing, axis=1).head()) #axis=1 defines that function is to be applied on each row



#Since ~86% values are “No”, it is safe to impute the missing values as “No” 
df['Self_Employed'].fillna('No',inplace=True)
df_test['Self_Employed'].fillna('No',inplace=True)


## Gender
#df['Gender'].fillna('Male',inplace=True)
#df_test['Gender'].fillna('Male',inplace=True)
 
#df['Gender'] =   df[ApplicantIncome].map(lambda x: "Female" if < 4643 lese "Male")
df['Gender'][df['Gender'].isnull() & df['ApplicantIncome'] < 4643] = 'Female'
df['Gender'][df['Gender'].isnull() & df['ApplicantIncome'] >= 4643] = 'Male'

df_test['Gender'][df_test['Gender'].isnull() & df_test['ApplicantIncome'] < 4643] = 'Female'
df_test['Gender'][df_test['Gender'].isnull() & df_test['ApplicantIncome'] >= 4643] = 'Male'

## credit history
#df['Credit_History'].fillna('1',inplace=True)#
#df_test['Credit_History'].fillna('1',inplace=True)

mask = (df['Credit_History'].isnull()) & (df['Loan_Status'] == 'N')
df.ix[mask, 'Credit_History'] = 0.0

mask = (df['Credit_History'].isnull()) & (df['Loan_Status'] == 'Y')
df.ix[mask, 'Credit_History'] = 1.0

#df['Credit_History'][df['Credit_History'].isnull() & df['Loan_Status'] == 'N'] = 0.0
#df['Credit_History'][df['Credit_History'].isnull() & df['Loan_Status'] == 'Y'] = 1.0

df_test['Credit_History'].fillna('1',inplace=True)



df['Married'].fillna('Yes',inplace=True)
df_test['Married'].fillna('Yes',inplace=True)


df['Dependents'].fillna('0',inplace=True)
df_test['Dependents'].fillna('0',inplace=True)

# fill NaN loanAmount
#df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)


table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
df_test['LoanAmount'].fillna(df_test[df_test['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


#df.dropna(subset=['Loan_Amount_Term'], axis=0, inplace=True)
df['Loan_Amount_Term'].fillna(df[df['Loan_Amount_Term'].isnull()].apply(fage, axis=1), inplace=True)
df_test['Loan_Amount_Term'].fillna(df_test[df_test['Loan_Amount_Term'].isnull()].apply(fage, axis=1), inplace=True)


df.apply(lambda x: sum(x.isnull()),axis=0) 

# use log transformation to nullify their effect:
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df_test['LoanAmount_log'] = np.log(df_test['LoanAmount'])
#df['LoanAmount_log'].hist(bins=20)

# add now column 'TotalIncome'
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
#df['LoanAmount_log'].hist(bins=20) 
df_test['TotalIncome'] = df_test['ApplicantIncome'] + df_test['CoapplicantIncome']
df_test['TotalIncome_log'] = np.log(df_test['TotalIncome'])

df['LoanAmountPerIncome'] = df['LoanAmount']/df['TotalIncome']
df_test['LoanAmountPerIncome'] = df_test['LoanAmount']/df_test['TotalIncome']


#Applying per column:
print("Missing values per column (after fill in NaN):")
print(df.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column
print(df_test.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

#sys.exit()

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
    if ( i != 'Loan_Status'):
        df_test[i] = le.fit_transform(df_test[i])
df.dtypes 



#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 
  

df = df.drop(labels=['Loan_ID'], axis=1) 

df_test_Loan_ID =  df_test.loc[:, ['Loan_ID']]
df_test = df_test.drop(labels=['Loan_ID'], axis=1) 

outcome_var = 'Loan_Status'

#normalize numericalcontinuous feature

cols_to_normalize = [\
        'Loan_Amount_Term'\
       ,'LoanAmountPerIncome'\
       ,'TotalIncome_log'\
       ,'LoanAmount_log'\
]

T = preprocessing.StandardScaler().fit(df.loc[:, cols_to_normalize])
df.loc[:, cols_to_normalize] = T.transform(df.loc[:,cols_to_normalize])
df_test.loc[:, cols_to_normalize] = T.transform(df_test.loc[:,cols_to_normalize])


#model = LogisticRegression()
#predictor_var = ['Credit_History']
#classification_model(model, df, predictor_var, outcome_var)

##We can try different combination of variables:
#predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
#classification_model(model, df, predictor_var, outcome_var)


model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features='auto')
predictor_var = [\
                  'Credit_History'\
                 ,'LoanAmountPerIncome'\
                 ,'TotalIncome_log'\
                 ,'LoanAmount_log'\
                 ,'Loan_Amount_Term'\
                 ,'Property_Area'\
#                 ,'Married'\
#                 ,'Education'\
#                 ,'Dependents'\
                  ,'Gender'\
#                 ,'Self_Employed'\
                 ]
classification_model(model, df,predictor_var,outcome_var)

#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)

 

model.predict(df_test[predictor_var])

predict_Loan_Status = pd.DataFrame(model.predict(df_test[predictor_var]), columns=['Loan_Status'])

predict_result = pd.concat([df_test_Loan_ID['Loan_ID'], predict_Loan_Status], axis = 1)



predict_result.to_csv(r"C:\Data Science Projects\Load Prediction\predict_result4.csv", sep=',', header=True, decimal='.', index =False)

#print(predict_Loan_Status)