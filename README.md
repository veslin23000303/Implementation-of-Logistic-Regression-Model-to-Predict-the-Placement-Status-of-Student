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
Developed by: VESLIN ANISH A
RegisterNumber:  212223240175
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
![Screenshot 2024-09-20 103142](https://github.com/user-attachments/assets/98dc7376-e703-4b12-aa1a-2182ab9b96a2)
![Screenshot 2024-09-20 103148](https://github.com/user-attachments/assets/ed111c55-6674-483a-a315-256d0d985f16)
![Screenshot 2024-09-20 103217](https://github.com/user-attachments/assets/c1e32067-eab8-446d-b388-d59ddaeb461c)
![Screenshot 2024-09-20 103238](https://github.com/user-attachments/assets/10aa5cf5-38bd-4a40-9374-cb5b1f096826)
![Screenshot 2024-09-20 103249](https://github.com/user-attachments/assets/4a722947-4a09-4494-b7ec-3e7ea4b1a507)
![Screenshot 2024-09-20 103258](https://github.com/user-attachments/assets/761afe6b-3585-447f-9360-6b3b4676ef60)
![Screenshot 2024-09-20 103305](https://github.com/user-attachments/assets/6ad2f896-642d-498a-a46e-6e3313af8ba3)
![Screenshot 2024-09-20 103309](https://github.com/user-attachments/assets/f317a9c7-acc6-4d88-af72-d0b8f3d55c31)








## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
