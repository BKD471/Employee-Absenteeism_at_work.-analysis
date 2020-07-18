# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:15:42 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Absenteeism_at_workex.csv')
df=dataset.copy()

df.info()



df.isnull().sum(axis=0)


df=df.drop(['ID'],axis=1)



len(df['Reason for absence'].unique())
df['Absenteeism time in hours'].unique()
len(df['Absenteeism time in hours'].unique())
df['Service time'].unique()
len(df['Service time'].unique())
df['Age'].unique()
len(df['Age'].unique())

sorted(df['Reason for absence'].unique())


#working on reason for absence dummies

reason_col=pd.get_dummies(df['Reason for absence'])
reason_col

#performing check on the correctness of reason_col
reason_col['check']=reason_col.sum(axis=1)
reason_col
reason_col['check'].sum(axis=0)

#discarding check;
reason_col=reason_col.drop(['check'],axis=1)


reason_col=pd.get_dummies(df['Reason for absence'],drop_first=True)
reason_col


df.columns.values

reason_col.columns.values

df=df.drop(['Reason for absence'],axis=1)


reason_type1=reason_col.loc[:,1:14].max(axis=1)
reason_type2=reason_col.loc[:,15:17].max(axis=1)
reason_type3=reason_col.loc[:,18:21].max(axis=1)
reason_type4=reason_col.loc[:,22:].max(axis=1)



df=pd.concat([df,reason_type1,reason_type2,reason_type3,reason_type4],axis=1)


df.columns.values
column_names=['Month of absence', 'Day of the week', 'Seasons',
       'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours', 'reason_type1', 'reason_type2', 'reason_type3','reason_type4']
df.columns=column_names
df.columns.values

col_reorder=['reason_type1', 'reason_type2',
       'reason_type3', 'reason_type4','Month of absence', 'Day of the week', 'Seasons',
       'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target',
       'Disciplinary failure', 'Education', 'Son', 'Social drinker',
       'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours']
df=df[col_reorder]


df['Education'].unique()
#1-->high school 2-->graduate  3-->post graduate 4-->doctor/master

df['Education'].value_counts()

df['Education']=df['Education'].map({1:0,2:1,3:1,4:1})

df['Education'].unique()
df['Education'].value_counts()


df_proc=df.copy()