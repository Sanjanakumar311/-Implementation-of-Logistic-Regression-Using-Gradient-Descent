# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights and bias, then compute predictions using the sigmoid function.

2. Calculate the cost (log loss) and gradients for weights and bias.

3. Update parameters iteratively using gradient descent until convergence.

4. Predict outcomes, evaluate accuracy, and visualize cost convergence and decision boundary.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SANJANA K L
RegisterNumber:  212224230241
*/
```

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.cost_history = []

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.W) + self.b
            y_pred = sigmoid(linear_model)

            cost = -(1/self.m) * np.sum(y*np.log(y_pred+1e-9) + (1-y)*np.log(1-y_pred+1e-9))
            self.cost_history.append(cost)

            dw = (1/self.m) * np.dot(X.T, (y_pred - y))
            db = (1/self.m) * np.sum(y_pred - y)

            self.W -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.W) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
X = np.array([
    [8.5, 120], [6.2, 100], [7.8, 110], [5.5, 85],
    [9.0, 130], [6.0, 95], [7.0, 105], [8.8, 125]
])
y = np.array([1, 0, 1, 0, 1, 0, 1, 1])

X = (X - X.mean(axis=0)) / X.std(axis=0)

model = LogisticRegressionGD(lr=0.1, epochs=2000)
model.fit(X, y)

y_pred = model.predict(X)
print("Predictions:", y_pred)
print("Actual     :", y)
print("Accuracy   :", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

plt.plot(model.cost_history)
plt.title("Cost Function Convergence")
plt.xlabel("Epochs")
plt.ylabel("Cost (Log Loss)")
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolors="k")
x1 = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2 = -(model.W[0]*x1 + model.b) / model.W[1]  # boundary equation
plt.plot(x1, x2, color="green", label="Decision Boundary")
plt.legend()
plt.title("Decision Boundary for Logistic Regression (GD)")
plt.xlabel("Feature 1 (scaled CGPA)")
plt.ylabel("Feature 2 (scaled IQ)")
plt.show()

```

## Output:
<img width="574" height="92" alt="image" src="https://github.com/user-attachments/assets/0297927c-b1fa-4dda-82f6-71a16616e65d" />


<img width="622" height="137" alt="image" src="https://github.com/user-attachments/assets/9b7fe1a4-7322-4c53-ac5d-f0441e868d6f" />


<img width="1109" height="298" alt="image" src="https://github.com/user-attachments/assets/26ac4dc7-6348-4174-8dc7-6ea6f8aad089" />


<img width="1048" height="559" alt="image" src="https://github.com/user-attachments/assets/033586e1-1877-4890-8131-d72e2b943046" />


<img width="990" height="556" alt="image" src="https://github.com/user-attachments/assets/0fa99d4e-360f-45e0-bd75-7412c78cf51e" />


# Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

