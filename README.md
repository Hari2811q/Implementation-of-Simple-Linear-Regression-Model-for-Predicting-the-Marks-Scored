# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hariprasath R
RegisterNumber: 212223040059
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SMARTLINK/Downloads/student_scores.csv")
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#spilitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![image](https://github.com/user-attachments/assets/712d82d4-388e-48b1-834a-46c79d207da0)
![image](https://github.com/user-attachments/assets/bb4a5f66-bd38-4a81-840b-d75e240aa442)
![image](https://github.com/user-attachments/assets/b7c2ad4a-a70f-4a0c-abac-3ddfaf7c257c)
![image](https://github.com/user-attachments/assets/15431746-6d6d-4cb4-8b48-5ca75c155fa7)
![image](https://github.com/user-attachments/assets/0fb8888c-bba4-452d-8a85-c5b964ed2394)
![image](https://github.com/user-attachments/assets/a5afa76f-0117-4ff4-8e3e-f8c37ab779d6)
![image](https://github.com/user-attachments/assets/c64773ea-b5d7-45a1-a77b-48e5e9ae0fdd)
![image](https://github.com/user-attachments/assets/fa04c6ec-0a74-4364-a4ea-e2e8d01eb851)
![image](https://github.com/user-attachments/assets/97374d40-3691-499c-8ce7-fbccdb5a3d36)
![image](https://github.com/user-attachments/assets/5f309c86-569b-4c39-aa36-c3009f3fb7b5)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
