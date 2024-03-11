# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the python library pandas
2.Read the dataset of Placement_Data
3.Copy the dataset in data1
4.Remove the columns which have null values using drop()
5.Import the LabelEncoder for preprocessing of the dataset
6.Assign x and y as status column values
7.From sklearn library select the model to perform Logistic Regression
8.Print the accuracy, confusion matrix and classification report of the dataset

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PULI NAGA NEERAJ
RegisterNumber:  212223240130
*/
```
```
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()
```
```
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1) # Removes the specified row or column
data1.head()
```
```
data1.isnull().sum()
data1.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
```
```
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
```
```
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver ='liblinear') # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
```
```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) # Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) ,True +ve/
#accuracy_score (y_true,y_pred, normalize = false)
# Normalize : It contains the boolean value (True/False). If False, return the number of correct
accuracy
```
```
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
```
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![ML-4 1](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/67063aae-b0e0-4594-b2fd-7b439427a646)
![ML-4 2](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/1098eec3-1710-4fe1-a88e-9f22b596cb9a)
![ML-4 3](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/f8b05009-c740-4e81-9fb9-8ada95355687)
![ML-4 4](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/326a3a85-4c44-4b3e-8c0c-c889d96e550c)
![ML-4 5](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/1e98bf2e-7e67-4100-8f1f-b43dabd9d018)
![ML-4 6](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/98d6a399-86b0-4df3-95e1-c3fe38a78c73)
![ML-4 6](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/b2e9c27c-895d-4d67-9535-bbba7f46491a)
![ML-4 6](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/974b6a0e-9610-477f-bea4-66aa907a7cf0)
![ML-4 6](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/7456f1d4-1c8e-47a7-b41b-b15b7da566e1)
![ML-4 8](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/ab97b1d0-605e-48a8-824f-676f78f84666)
![ML-4 9](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/9090e09b-e103-449f-ba4b-9ab24ba7e50b)
![ml-4 10](https://github.com/PuliNagaNeeraj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849173/125fbfdc-cf85-47f1-bb52-d49ebd3cf15a)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
