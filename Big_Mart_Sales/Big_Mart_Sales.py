# -*- coding: utf-8 -*-
"""
Created on Nov 01 07:10:06 2018

@author: Joe

"""

import pandas as pd
import numpy as np
import sys, os


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)

df_train = pd.read_csv(r"C:\Data Science Projects\Big mart Sales\Data\train_data.csv", sep=',', header=0, decimal='.')
df_test = pd.read_csv(r"C:\Data Science Projects\Big mart Sales\Data\test_data.csv", sep=',', header=0, decimal='.')


df_train['source']='train'
df_test['source']='test'

df_data = pd.concat([df_train, df_test],ignore_index=True, sort=False)
print(df_train.shape, df_test.shape, df_data.shape)
print("\n")

print(df_data.head(20))
print("\n")
print(df_data.dtypes)
print("\n")
print(df_data.describe())

#sys.exit()


#check NaN column
print("\n")
print(df_data.apply(lambda x: sum(x.isnull())))

#print(df_data.apply(lambda x: len(x.unique())).sort_values())


#Change categories of low fat:
print("\n")
print('Original Categories:')
print(df_data['Item_Fat_Content'].value_counts())

print('\nModified Categories:')
df_data['Item_Fat_Content'] = df_data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print(df_data['Item_Fat_Content'].value_counts())

#sys.exit()

#combine Item_Identifier feature 
#Get the first two characters of ID:
df_data['Item_Type_Combined'] = df_data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
df_data['Item_Type_Combined'] = df_data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
print("\n")
print(df_data['Item_Type_Combined'].value_counts())


#Mark non-consumables as separate category in low_fat:
df_data.loc[df_data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
print("\n")
print(df_data['Item_Fat_Content'].value_counts())

#sys.exit()



print("\n")
#Determine the average weight per item:
item_avg_weight = df_data.pivot_table(values='Item_Weight', index='Item_Identifier')
#Get a boolean variable specifying missing Item_Weight values
miss_bool = df_data['Item_Weight'].isnull() 
print('(Item_Weight) Orignal #missing: %d'% sum(miss_bool))
#Impute data and check #missing values before and after imputation to confirm
df_data.loc[miss_bool,'Item_Weight'] = df_data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
#df_data['Item_Weight'].fillna(df_data.groupby(by=['Item_Identifier'] )['Item_Weight'].transform("mean"), inplace=True) 
print('(Item_Weight) Final #missing: %d'% sum(df_data['Item_Weight'].isnull()))
      

print("\n")      
#Get a boolean variable specifying missing Outlet_Size values
miss_bool = df_data['Outlet_Size'].isnull() 
#Impute data and check #missing values before and after imputation to confirm
print('(Outlet_Size) Orignal #missing: %d'% sum(miss_bool))
#df_data.loc[miss_bool,'Outlet_Size'] = df_data.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
df_data['Outlet_Size'] = df_data.groupby('Outlet_Type').Outlet_Size.transform(lambda x: x.fillna(x.mode()[0]))
print('(Outlet_Size) Final #missing: %d', sum(df_data['Outlet_Size'].isnull()))      
      
      
#Get a boolean variable specifying 0 Item_Visibility values
miss_bool = (df_data['Item_Visibility'] == 0)
print('Orignal #missing: %d'% sum(miss_bool))

df_data['Item_Visibility'].fillna(df_data.groupby(by=['Item_Identifier'] )['Item_Visibility'].transform("mean"), inplace=True) 

print('Final #missing: %d'% sum(df_data['Item_Visibility'].isnull()))      

#sys.exit()      


#Determine another variable with means ratio
print("\n")
visibility_avg = pd.pivot_table(df_data, index=['Item_Identifier'], values=['Item_Visibility'], aggfunc=[np.mean])
df_data['Item_Visibility_MeanRatio'] = df_data.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']], axis=1)
print(df_data['Item_Visibility_MeanRatio'].describe())

#print(df_data.pivot_table(values='Item_Outlet_Sales',index='Outlet_Type'))

print("\n")
#Years:
df_data['Outlet_Years'] = 2018 - df_data['Outlet_Establishment_Year']
print(df_data['Outlet_Years'].describe())

#sys.exit()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
df_data['Outlet'] = le.fit_transform(df_data['Outlet_Identifier'])
cat_cols = ['Item_Fat_Content'\
,'Outlet_Location_Type'\
,'Outlet_Size'\
,'Item_Type_Combined'\
,'Outlet_Type'\
,'Outlet'
]

le = LabelEncoder()
for col in cat_cols:
    df_data[col] = le.fit_transform(df_data[col])


print(df_data.head(5))

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
corr = df_data.select_dtypes(include = ['float64', 'int64']).iloc[:, :].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, vmax=.8, square=True)
plt.show()

print("\n")
corr = df_data.select_dtypes(include = ['float64', 'int64']).iloc[:, :].corr()
cor_dict = corr['Item_Outlet_Sales'].to_dict()
del cor_dict['Item_Outlet_Sales']
print("List the numerical features decendingly by their correlation with Item_Outlet_Sales:\n")
for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
    print("{0}: \t{1}".format(*ele))

#sys.exit()    
    
#One Hot Coding:
df_data = pd.get_dummies(df_data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])    

print(df_data.head(5))      


#Drop the columns which have been converted to different types:
print("\n")

df_data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)
#Divide into test and train:
train = df_data.loc[df_data['source']=="train"]
test = df_data.loc[df_data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

print(df_data.head(5)) 
#sys.exit()    


#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']

from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print( "CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)



#from sklearn.linear_model import LinearRegression, Ridge, Lasso
#predictors = [x for x in train.columns if x not in [target]+IDcol]
## print predictors
#alg_liner = LinearRegression(normalize=True)
#modelfit(alg_liner, train, test, predictors, target, IDcol, 'alg_liner.csv')
#coef1 = pd.Series(alg_liner.coef_, predictors).sort_values()
#coef1.plot(kind='bar', title='Model Coefficients')
#plt.show()



#from sklearn.tree import DecisionTreeRegressor
#predictors = [x for x in train.columns if x not in [target]+IDcol]
#alg_dTree = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
#modelfit(alg_dTree, train, test, predictors, target, IDcol, 'alg_dTree.csv')
#coef_dTree = pd.Series(alg_dTree.feature_importances_, predictors).sort_values(ascending=False)
#coef_dTree.plot(kind='bar', title='Feature Importances')
#plt.show()

#random forest
from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg_RndmForest = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg_RndmForest, train, test, predictors, target, IDcol, 'alg_RndmForest.csv')
coef_RndmForest = pd.Series(alg_RndmForest.feature_importances_, predictors).sort_values(ascending=False)
coef_RndmForest.plot(kind='bar', title='Feature Importances')
plt.show()



from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg_RndmForest_2 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg_RndmForest_2, train, test, predictors, target, IDcol, 'alg_RndmForest_2.csv')
coef_RndmForest_2 = pd.Series(alg_RndmForest_2.feature_importances_, predictors).sort_values(ascending=False)
coef_RndmForest_2.plot(kind='bar', title='Feature Importances')
plt.show()
#sys.exit()    


#model.predict(df_test[predictor_var])
#predict_Loan_Status = pd.DataFrame(model.predict(df_test[predictor_var]), columns=['Loan_Status'])
#predict_result = pd.concat([df_test_Loan_ID['Loan_ID'], predict_Loan_Status], axis = 1)
#predict_result.to_csv(r"C:\Data Science Projects\Load Prediction\predict_result4.csv", sep=',', header=True, decimal='.', index =False)

#print(predict_Loan_Status)