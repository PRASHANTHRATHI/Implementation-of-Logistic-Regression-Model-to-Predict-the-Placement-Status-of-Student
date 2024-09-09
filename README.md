# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Prashanth.K
RegisterNumber:  212223230152
*/
```
```
import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

# Head:
![image](https://github.com/user-attachments/assets/1efcc686-7eab-4cd4-b9c1-0d796c896970)


# Data Copy:
![image](https://github.com/user-attachments/assets/5be34e0a-f570-4ee9-bae8-397949b419ef)


# Fit Transform:
![image](https://github.com/user-attachments/assets/cd8da353-7c20-4b3f-89ad-01e4c50a1a30)


# Logistic Regression:
![image](https://github.com/user-attachments/assets/3fbdb25e-f2bb-4e3b-b923-6f6d4160071b)

# Accuracy Score:
![image](https://github.com/user-attachments/assets/3f8aa101-3aa1-47fd-b67a-308ab84f0241)

# Confusion Matrix:
![image](https://github.com/user-attachments/assets/c9349b35-b1f5-40cc-8f01-7ed2ebed78f7)

# Classification Report:
![image](https://github.com/user-attachments/assets/509d1b75-62ab-4cbb-9bb4-2bd2d1c104a4)

# Prediction:
![image](https://github.com/user-attachments/assets/0221633c-d495-4e7d-a46f-18ace64a6931)











## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
