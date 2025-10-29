# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the Logistic Regression Using Gradient Descent.

## Equipments Required:

1. Hardware – PCs
  
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data.

2. Split Dataset into Training and Testing Sets.

3. Train the Model Using Stochastic Gradient Descent (SGD).

4. Make Predictions and Evaluate Accuracy.

5. Generate Confusion Matrix.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Lorena Avelyn R
RegisterNumber: 212224040174 
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)  

```

## Output:
<img width="976" height="321" alt="image" src="https://github.com/user-attachments/assets/d5c0b561-ab5f-4fc0-8786-965759bda25f" />
<img width="1076" height="69" alt="image" src="https://github.com/user-attachments/assets/65a61d88-7d2f-4cde-a4af-e760180dc3bb" />
<img width="516" height="129" alt="image" src="https://github.com/user-attachments/assets/076fcddf-316c-4296-88ad-cfa80af57d5d" />




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

