# ============================================================
# STEP 1: Import Libraries
# ============================================================
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# STEP 2: Load Dataset
# ============================================================
df = pd.read_csv("train_u6lujuX_CVtuZ9i (1) (1).csv")  # <-- put your CSV in the same folder
print("Shape:", df.shape)
print(df.head())


# ============================================================
# STEP 3: Handle Missing Values (explicit per column)
# ============================================================
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

print("\nNull values after handling:")
print(df.isnull().sum())


# ============================================================
# STEP 4: Encode Categorical Columns
# ============================================================
label = LabelEncoder()

df['Gender']        = label.fit_transform(df['Gender'])         # Male=1, Female=0
df['Married']       = label.fit_transform(df['Married'])        # Yes=1, No=0
df['Education']     = label.fit_transform(df['Education'])      # Graduate=0, Not Graduate=1
df['Self_Employed'] = label.fit_transform(df['Self_Employed'])  # Yes=1, No=0
df['Loan_Status']   = label.fit_transform(df['Loan_Status'])    # Y=1, N=0

# Fix 3+ in Dependents
df = df.replace(to_replace='3+', value=4)
df['Dependents'] = df['Dependents'].astype(int)

# Property Area manual mapping
df.replace({'Property_Area': {'Semiurban': 0, 'Urban': 1, 'Rural': 2}}, inplace=True)

print("\nData after encoding:")
print(df.head())


# ============================================================
# STEP 5: Visualization (optional, comment out if not needed)
# ============================================================
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.title("Education vs Loan Status")
plt.savefig("education_vs_loan.png")
plt.clf()

sns.countplot(x='Gender', hue='Loan_Status', data=df)
plt.title("Gender vs Loan Status")
plt.savefig("gender_vs_loan.png")
plt.clf()


# ============================================================
# STEP 6: Separate Features and Target
# ============================================================
X = df.drop(columns=['Loan_ID', 'Loan_Status'])
y = df['Loan_Status']


# ============================================================
# STEP 7: Scale Numeric Columns (ONE scaler, used everywhere)
# ============================================================
numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print("\nFeatures after scaling:")
print(X.head())


# ============================================================
# STEP 8: Train/Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")


# ============================================================
# STEP 9: Compare Multiple Models
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42)
}

print("\n--- Model Comparison ---")
best_model = None
best_acc = 0

for name, m in models.items():
    m.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, m.predict(X_train))
    test_acc  = accuracy_score(y_test,  m.predict(X_test))
    print(f"{name:25s} | Train: {train_acc*100:.2f}%  | Test: {test_acc*100:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        best_model = m
        best_name = name

print(f"\nBest model: {best_name} with test accuracy: {best_acc*100:.2f}%")


# ============================================================
# STEP 10: Detailed Evaluation of Best Model
# ============================================================
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Approved", "Approved"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# STEP 11: Save Model and Scaler
# ============================================================
joblib.dump(best_model, 'loan_model.pkl')
joblib.dump(scaler,     'scaler.pkl')
print("\nModel and scaler saved successfully!")


# ============================================================
# STEP 12: Test with a Single Sample
# ============================================================
# Format: Gender, Married, Dependents, Education, Self_Employed,
#         ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
#         Credit_History, Property_Area

sample = np.array([[1, 0, 0, 1, 0, 1853, 2840, 114, 360, 1, 0]])

# Scale only the numeric columns (indices 5,6,7,8)
sample[:, [5, 6, 7, 8]] = scaler.transform(sample[:, [5, 6, 7, 8]])

prediction = best_model.predict(sample)
print("\nSample Prediction:", "Loan Approved ✅" if prediction[0] == 1 else "Loan Not Approved ❌")