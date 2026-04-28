import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("🚀 Training started...")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/loan_data.csv")

# Remove unwanted column
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

print("✅ Data loaded")

# ----------------------------
# CLEANING
# ----------------------------
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

# Fix Dependents
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Income_per_Loan'] = df['TotalIncome'] / df['LoanAmount']

# ----------------------------
# SPLIT
# ----------------------------
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# ----------------------------
# TRAIN TEST SPLIT (FIRST!)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# COLUMN TYPES
# ----------------------------
num_cols = [
    'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'TotalIncome', 'Income_per_Loan'
]

cat_cols = [
    'Gender', 'Married', 'Dependents',
    'Education', 'Self_Employed', 'Property_Area'
]

# ----------------------------
# PREPROCESSOR
# ----------------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# ----------------------------
# MODEL COMPARISON
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "SVM": SVC(probability=True)
}

print("\n🔍 Model Comparison (Cross Validation):\n")

for name, model in models.items():

    pipe = Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])

    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.2f}")

# ----------------------------
# HYPERPARAMETER TUNING
# ----------------------------
print("\n🔧 Hyperparameter Tuning (Random Forest)...")

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [5, 10, None],
    'model__min_samples_split': [2, 5]
}

grid = GridSearchCV(
    Pipeline([
        ('preprocessing', preprocessor),
        ('model', RandomForestClassifier(random_state=42))
    ]),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n🏆 Best Parameters:")
print(grid.best_params_)

print(f"🔥 Best Score: {grid.best_score_:.2f}")

# ----------------------------
# FINAL MODEL
# ----------------------------
pipeline = grid.best_estimator_

# ----------------------------
# EVALUATE
# ----------------------------
preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\n✅ Accuracy: {acc:.2f}")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, preds))

print("\n📋 Classification Report:")
print(classification_report(y_test, preds))

# ----------------------------
# PROBABILITY OUTPUT
# ----------------------------
probs = pipeline.predict_proba(X_test)
print("\n🔮 Sample Probabilities:")
print(probs[:5])

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(pipeline, "loan_model.pkl")

print("\n💾 Model saved as loan_model.pkl")