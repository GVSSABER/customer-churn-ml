# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_excel("C:/Users/garla/OneDrive/Desktop/churn prediction/churnprediction.xlsx")

# ==============================
# 3. Check Data
# ==============================
print("First 5 rows:")
print(df.head())

print("\nChurn Distribution:")
print(df['Churn Value'].value_counts())

# ==============================
# 4. Handle Missing Values
# ==============================
df.fillna(df.mean(numeric_only=True), inplace=True)

# ==============================
# 5. Drop Unnecessary Columns
# ==============================
df = df.drop(['CustomerID', 'Churn Reason'], axis=1, errors='ignore')

# ==============================
# 6. Define Features (X) and Target (y)
# ==============================
y = df['Churn Value']
X = df.drop(['Churn Value', 'Churn Label'], axis=1)

# ==============================
# 7. Convert Categorical to Numerical
# ==============================
X = pd.get_dummies(X, drop_first=True)

# ==============================
# 8. Split Data (Train & Test)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 9. Train Model
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# 10. Make Predictions
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 11. Evaluate Model
# ==============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nModel Comparison:")
print("Logistic Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
import pandas as pd
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()