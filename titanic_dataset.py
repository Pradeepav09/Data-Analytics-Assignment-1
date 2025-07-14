# titanic_case_study.py

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, classification_report

# Step 2: Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Dataset Loaded. Shape:", df.shape)

# Step 3: Preprocessing
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'] = le.fit_transform(df['Embarked'])
df['Age'] = df['Age'].fillna(df['Age'].median())

# Step 4: Visualization
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.show()

sns.countplot(data=df, x='Survived', hue='Sex')
plt.title("Survival by Gender")
plt.show()

sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

# Step 5: Model Building
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Step 6: Evaluation
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Cross-Entropy Error (Log Loss):", log_loss(y_test, y_prob))

# Step 7: 10-Fold Cross Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print("10-Fold CV Accuracy Scores:", cv_scores)
print("Average Accuracy:", cv_scores.mean())

# Step 8: Training with Different Split Ratios
def train_with_split(ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-ratio), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Train/Test Split = {int(ratio*100)}/{int((1-ratio)*100)} --> Accuracy = {acc:.4f}")

for ratio in [0.6, 0.7, 0.8, 0.9]:
    train_with_split(ratio)
