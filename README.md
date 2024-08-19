# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KUSHALI PG
RegisterNumber:212223230110
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ALLEN JOVETH P
RegisterNumber: 212223240007
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b 
*/
```

## Output:
![image](https://github.com/user-attachments/assets/7d126d17-46f4-4ffd-9ebe-5274f9e68aac)

![image](https://github.com/user-attachments/assets/b03b3141-5294-45b9-928f-f87fbe8cf4c7)

![image](https://github.com/user-attachments/assets/63fdacbc-8c3e-4bea-8d86-89928a9574d1)

![image](https://github.com/user-attachments/assets/dd456cff-b888-448b-a74a-6f392aab5b5a)

![image](https://github.com/user-attachments/assets/ba3330c3-6a17-48e5-b867-c40a9d56be46)

![image](https://github.com/user-attachments/assets/8e5e6cbd-d6bd-4071-8d57-f61ef0b183a7)

![image](https://github.com/user-attachments/assets/8ac84cfb-e3fd-4217-9aa6-016168179001)

![image](https://github.com/user-attachments/assets/7ed8cbb6-e806-43dc-b6c9-b10da14b8271)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
