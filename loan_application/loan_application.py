# -*- coding: utf-8 -*-
"""
Created on Dec 16 2017

@author: Joe

revision history

--(v3  score: 0.77778)
-- two features
                        Model  Score
2         Logistic Regression  83.22
0     Support Vector Machines  83.06
1                         KNN  83.06
3               Random Forest  83.06
4                 Naive Bayes  83.06
6  Stochastic Gradient Decent  83.06
7                  Linear SVC  83.06
8               Decision Tree  83.06
5                  Perceptron  40.07

--(v4 score: 0.77083)
--added 'TotalIncome_log', 'LoanAmount_log' 
                        Model   Score
3               Random Forest  100.00
8               Decision Tree  100.00
1                         KNN   86.48
2         Logistic Regression   83.22
0     Support Vector Machines   83.06
7                  Linear SVC   83.06
4                 Naive Bayes   82.74
6  Stochastic Gradient Decent   73.62
5                  Perceptron   64.33

-- (v5 score: 0.76389)
--added 'Loan_Amount_Term', 'Dependents' 
                       Model  Score
3               Random Forest  84.20
8               Decision Tree  84.20
2         Logistic Regression  83.22
0     Support Vector Machines  83.06
4                 Naive Bayes  83.06
7                  Linear SVC  83.06
6  Stochastic Gradient Decent  80.62
1                         KNN  53.58
5                  Perceptron  31.60


-- (v6   score:0.722222)
--added 'LoanAmount_perIncome', 'Dependents' 
                       Model  Score
3               Random Forest  99.19
8               Decision Tree  99.19
1                         KNN  85.18
4                 Naive Bayes  83.55
2         Logistic Regression  83.22
0     Support Vector Machines  83.06
7                  Linear SVC  83.06
5                  Perceptron  81.76
6  Stochastic Gradient Decent  55.21

-- (v7 score: 0.79167)
--added all                                
                       Model   Score
3               Random Forest  100.00
8               Decision Tree  100.00
1                         KNN   85.67
4                 Naive Bayes   83.39
2         Logistic Regression   83.22
0     Support Vector Machines   83.06
7                  Linear SVC   83.06
5                  Perceptron   76.22
6  Stochastic Gradient Decent   72.80

--LoanAmount use different way (mean of group grouped by Gender, Married, Self_Employed)  to fill NaN  (v8 0.77778 down)
                        Model   Score
3               Random Forest  100.00
8               Decision Tree  100.00
1                         KNN   84.20
4                 Naive Bayes   83.39
0     Support Vector Machines   83.06
2         Logistic Regression   83.06
7                  Linear SVC   83.06
6  Stochastic Gradient Decent   82.57
5                  Perceptron   77.52

-- (v9 score: 0.798611111111111)
use GridSearchCV   
----------------------------------------------------

Rank:  175 out of 3596

""" 
              

import pandas as pd
import numpy as np
import sys, os
from sklearn import preprocessing

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.grid_search import GridSearchCV


pd.set_option('display.max_rows', 1000)

#load the data

df_train = pd.read_csv(r"C:\Data Science Projects\Load Prediction\loan_train_data.csv", sep=',', header=0, decimal='.')
df_test = pd.read_csv(r"C:\Data Science Projects\Load Prediction\loan_test_data.csv", sep=',', header=0, decimal='.')
df_full = [df_train, df_test]

print(df_train.head(10))
print(df_train.describe())
print(df_train.info())
print(df_train.columns)


# count NaN rows
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print("Missing values per column:")

for dataset in df_full:
    print(dataset.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column


# 'Married': set to 'Yes' for NaN value (use majarity value)
for dataset in df_full:
    dataset['Married'].fillna('Yes',inplace=True)


# 'Gender': set to 'Male' for NaN value (use majarity value)
for dataset in df_full:
    dataset['Gender'].fillna('Male',inplace=True)

#df['Gender'] =   df[ApplicantIncome].map(lambda x: "Female" if < 4643 lese "Male")
#df['Gender'][df['Gender'].isnull() & df['ApplicantIncome'] < 4643] = 'Female'
#df['Gender'][df['Gender'].isnull() & df['ApplicantIncome'] >= 4643] = 'Male'
#df_test['Gender'][df_test['Gender'].isnull() & df_test['ApplicantIncome'] < 4643] = 'Female'
#df_test['Gender'][df_test['Gender'].isnull() & df_test['ApplicantIncome'] >= 4643] = 'Male'


# 'Dependents':set to 0 for NaN value (use majarity value)
for dataset in df_full:
    dataset['Dependents'].fillna('0',inplace=True) 

# repalce string to number
for dataset in df_full:
    dataset['Dependents'].replace(['0','1','2', '3+'], [0,1,2,3], inplace=True)    
    
# 'Self_Employed':
#Since ~86% values are “No”, it is safe to impute the missing values as “No” 
for dataset in df_full:
    dataset['Self_Employed'].fillna('No',inplace=True)

# add feature 'TotalIncome'
for dataset in df_full:
    dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
    
 
# fill NaN 'loanAmount'
#for dataset in df_full:
#    dataset['LoanAmount'].fillna(dataset['TotalIncome'] *(dataset['LoanAmount'].median()/dataset['TotalIncome'].median()), inplace=True)
for dataset in df_full:
    impute_grps = dataset.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)


    #iterate only through rows with missing LoanAmount
    for i,row in dataset.loc[dataset['LoanAmount'].isnull(),:].iterrows():
      ind = tuple([row['Gender'],row['Married'],row['Self_Employed']])
      dataset.loc[i,'LoanAmount'] = impute_grps.loc[ind].values[0]

# fill Nan 'Loan_Amount_Term'
for dataset in df_full:
    dataset['Loan_Amount_Term'].fillna(360.0, inplace=True) 
    
   
# add feature LoanAmount/TotalIncome
for dataset in df_full:
    dataset['Ratio_Income_vs_LoanAmount'] = dataset['TotalIncome']/dataset['LoanAmount']

  

#sys.exit()  
  
#add feature 'family_size'
for dataset in df_full:
    dataset['family_size'] = 0
    dataset.loc[dataset['Married'] == 'No', ['family_size']] = 1
    dataset.loc[dataset['Married'] == 'Yes', ['family_size']] = 2
    dataset['family_size'] =  dataset['family_size'] + dataset['Dependents']
    
# add feature 'Income_Person'
for dataset in df_full:
    dataset['Income_Person'] = dataset['TotalIncome']/dataset['family_size']
    

#sys.exit()    

# credit history
mask = (df_train['Credit_History'].isnull()) & (df_train['Loan_Status'] == 'N')
df_train.loc[mask, 'Credit_History'] = 0.0
    
mask = (df_train['Credit_History'].isnull()) & (df_train['Loan_Status'] == 'Y')
df_train.loc[mask, 'Credit_History'] = 1.0


df_test['Credit_History'].fillna('1',inplace=True)


# use log transformation to nullify their effect:
for dataset in df_full:
    dataset['LoanAmount_log'] = np.log(dataset['LoanAmount'])
    dataset['TotalIncome_log'] = np.log(dataset['TotalIncome'])

# display NaN count to make sure there is no NaN values
for dataset in df_full:
    print(dataset.apply(lambda x: sum(x.isnull()),axis=0) )

#sys.exit()

# encoder categoricak features
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df_train[i] = le.fit_transform(df_train[i])
    if ( i != 'Loan_Status'):
        df_test[i] = le.fit_transform(df_test[i])
        
print(df_train.dtypes)


for dataset in df_full:
    print("----  Check NaN --------------")
    print(dataset.apply(num_missing, axis=0))

#sys.exit()

#normalize numericalcontinuous feature

from sklearn.preprocessing import Imputer

cols_to_normalize = [\
         'Loan_Amount_Term'\
       , 'ApplicantIncome'\
       , 'CoapplicantIncome'\
       , 'TotalIncome'\
       , 'Income_Person'\
       , 'LoanAmount'\
       , 'Ratio_Income_vs_LoanAmount'\
       , 'LoanAmount_log'\
       , 'TotalIncome_log'\

]

T = preprocessing.StandardScaler().fit(df_train.loc[:, cols_to_normalize])
df_train.loc[:, cols_to_normalize] = T.transform(df_train.loc[:,cols_to_normalize])
df_test.loc[:, cols_to_normalize] = T.transform(df_test.loc[:,cols_to_normalize])

#sys.exit()


# classifier comparison

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)



predictor_col = [\
 'Loan_ID'\
,'Credit_History'\
,'Gender'\
,'Ratio_Income_vs_LoanAmount'\
,'TotalIncome_log'\
,'LoanAmount_log'\
,'Loan_Amount_Term'\
,'Property_Area'\
,'Married'\
,'Education'\
,'Dependents'\
,'Self_Employed'\
 ]

#,'Loan_Status'\

df_train = pd.concat([df_train.loc[:, predictor_col], df_train.loc[:,'Loan_Status']], axis=1)
df_test = df_test.loc[:, predictor_col]


X_train = df_train.drop(["Loan_ID", "Loan_Status"], axis=1)
Y_train = df_train["Loan_Status"]
X_test  = df_test.drop("Loan_ID", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

#X = X.reset_index()
#y = y.reset_index()
#sys.exit()

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print('SVC:', acc_svc)

#k-Nearest Neighbors algorithm
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print('KNeighborsClassifier:', acc_knn)


##
acc = LogisticRegression()
acc.fit(X_train, Y_train)
Y_pred = acc.predict(X_test)
acc_log = round(acc.score(X_train, Y_train) * 100, 2)

print('Logistic Regression:', acc_log)


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print('GaussianNB:', acc_gaussian)

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print('Perceptron:', acc_perceptron)


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print('Linear SVC:', acc_linear_svc)

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
print('Stochastic Gradient Descent:', acc_sgd)


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('Decision Tree:', acc_decision_tree)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Random Forest:', acc_random_forest)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})

#display model score     
print(models.sort_values(by='Score', ascending=False))

#sys.exit()

# Random Forest
def run_GridSearch_RandomForestRegressor():
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [150, 180, 200], 'max_features': ["auto", "sqrt", "log2"], 'min_samples_leaf': [1,2], 'min_samples_split':[2]}
        # then try 6 (2×3) combinations with bootstrap set as False
        #,{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    
    forest_classifer = RandomForestClassifier(random_state=42)
#    
#   # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
    model = GridSearchCV(forest_classifer, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    
    model.fit(X_train, Y_train)
    print(model.best_estimator_)
    return model

       

# use random_forest as the model
   
#run_GridSearch_RandomForestRegressor()    

model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=180, n_jobs=1,
            oob_score=False, random_state=42, verbose=0, warm_start=False).fit(X_train, Y_train)


# model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
##           max_features="auto", max_leaf_nodes=None, min_impurity_split=1e-07,
##           min_samples_leaf=1, min_samples_split=2,
##           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
##           oob_score=False, random_state=42, verbose=0, warm_start=False).fit(X_train, y_train)
#
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Random Forest:', acc_random_forest)


submission = pd.DataFrame({
        "Loan_ID": df_test["Loan_ID"],
        "Loan_Status": Y_pred
    })
    
submission['Loan_Status'].replace([0, 1], ['N','Y'], inplace=True)        
print(submission.head(10))

submission.to_csv('C:\Data Science Projects\Load Prediction\submission_v11.csv', index=False)

