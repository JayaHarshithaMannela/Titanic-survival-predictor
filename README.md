# Titanic Survival Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Features](#project-features)
- [How to Set Up the Project](#how-to-set-up-the-project)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Building and Training the Model](#building-and-training-the-model)
- [Model Evaluation](#model-evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Usage](#usage)
- [Contributing](#contributing)

---

## Project Overview
This project implements a machine learning model to predict whether a passenger survived the Titanic disaster using various features such as age, sex, passenger class, and more.

---

## Dataset
The dataset used is from Kaggle's **Titanic: Machine Learning from Disaster** competition.

👉 [Titanic Dataset on Kaggle](https://www.kaggle.com/c/titanic/data)

### Data Folder Structure
Place the `train.csv` and `test.csv` files inside the `data/` folder or update the file paths in your scripts accordingly.

---

## Project Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, feature scaling.
- **Exploratory Data Analysis (EDA)**: Visualizing trends and relationships.
- **Model Training**: Logistic Regression, Random Forest, SVM, etc.
- **Model Evaluation**: Accuracy, precision, recall, F1-score.

---

## How to Set Up the Project

### Prerequisites
Make sure you have **Python 3.x** installed (recommended: 3.7+).

### Install Dependencies
If using a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

If not, manually install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### Step 2: Load the Dataset
Download the dataset from Kaggle and place the `train.csv` and `test.csv` files in the `data/` folder.

---

## Exploratory Data Analysis (EDA)

You can perform EDA to visualize trends and relationships. For example:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/train.csv')
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()
```

---

## Building and Training the Model

Example: Logistic Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('data/train.csv')
X = pd.get_dummies(df.drop(['Survived'], axis=1))
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

---

## Model Evaluation

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## Saving and Loading the Model

```python
import joblib

# Save
joblib.dump(model, 'models/titanic_model.pkl')

# Load
model = joblib.load('models/titanic_model.pkl')
predictions = model.predict(X_test)
```

---

## Usage
- Run `train.py` to train the model.
- Use `predict.py` to make predictions on new data.
- Use `evaluate.py` to assess model performance.

---

## Contributing
Feel free to fork the repository, make changes, and submit a pull request. Contributions are welcome!
