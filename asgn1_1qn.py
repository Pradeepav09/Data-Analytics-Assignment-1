import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('online_shoppers_intention.csv')
print("Sample Data:\n", df.head())

# STEP 3: Understand the Data
print("Shape:", df.shape)
df.info()
print("\nMissing Values:\n", df.isnull().sum())

# STEP 4: Data Cleaning
df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)

# STEP 5: Encode Categorical Columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# STEP 6: Feature Scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled, columns=df.columns)

# STEP 7: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# STEP 8: Split Data
X = df_scaled.drop("Revenue", axis=1)
y = df_scaled["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 9: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# STEP 10: Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 11: Cross Entropy
proba = model.predict_proba(X_test)
print("\nCross Entropy Loss:", log_loss(y_test, proba))

# STEP 12: 10-Fold Cross Validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv)
print("\n10-Fold CV Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# STEP 13: Accuracy in Various Train/Test Ratios
print("\nTrain/Test Ratios:")
for ratio in [0.2, 0.3, 0.4, 0.5]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test Size {ratio}: Accuracy = {score:.4f}")
