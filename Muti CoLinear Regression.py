# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:01:52 2021

@author: kavita.gaikwad
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")

X= dataset.iloc[:,:-1]
y = dataset.iloc[:,4]

states = pd.get_dummies(X['State'],drop_first=True)

## Drp the State column 
X = X.drop('State',axis=1)

X = pd.concat([X,states],axis=1)

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

## fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


## Predicting the Test set result 
y_pred = regressor.predict(X_test)


from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)




