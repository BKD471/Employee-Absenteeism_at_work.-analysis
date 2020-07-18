# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:52:58 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:42:28 2020

@author: User
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('absentism_processed.csv')
dataset.info()
dataset['Absenteeism time in hours'].median()


targets=np.where(dataset['Absenteeism time in hours']>3,1,0)

dataset['Excessive Absentism']=targets



targets.sum()/targets.shape[0]



data_with_targets=dataset.drop(['Absenteeism time in hours','Distance from Residence to Work','Hit target','Work load Average/day ','Education'],axis=1)




unscaled_inputs=data_with_targets.iloc[:,:-1]


from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
absent_scaler=StandardScaler()
class CustomScaler(BaseEstimator,TransformerMixin): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler(copy,with_mean,with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    
    # the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
    
    
    
unscaled_inputs.columns.values
#columns_to_scale = ['Month of absence', 'Day of the week', 'Seasons',
 #      'Transportation expense', 'Distance from Residence to Work',
  #     'Service time', 'Age', 'Work load Average/day ', 'Hit target',
  #     'Disciplinary failure', 'Son', 'Social drinker',
   #    'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index']    

columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]

absent_scaler = CustomScaler(columns_to_scale)
absent_scaler.fit(unscaled_inputs)
scaled_input=absent_scaler.transform(unscaled_inputs)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(scaled_input,targets,train_size=0.75,random_state=20)


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(metric='minkowski',p=2)
classifier.fit(X_train,y_train)

classifier.score(X_train,y_train)


y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)








classifier.score(X_test,y_test)

