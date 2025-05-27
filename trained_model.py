import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Create output directory
os.makedirs("trained_data", exist_ok=True)

# Load dataset
df = pd.read_csv(r"C:\Users\PUP-CITE-CAD08\Desktop\JOHN PAUL A\Embalsado_API_Backend\csv\Students_Grading_Dataset.csv")

# Create Pass_Fail based on Grade
passing_grades = ["A", "B", "C"]
df['Pass_Fail'] = df['Grade'].apply(lambda x: 1 if x in passing_grades else 0)

# Features excluding 'Grade' since it's now target source
features_to_use = [
    "Gender",
    "Age",
    "Department",
    "Attendance (%)",
    "Midterm_Score",
    "Final_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Participation_Score",
    "Projects_Score",
    "Total_Score",
    # "Grade",  # excluded!
    "Study_Hours_per_Week",
    "Extracurricular_Activities",
    "Internet_Access_at_Home",
    "Parent_Education_Level",
    "Family_Income_Level",
    "Stress_Level (1-10)",
    "Sleep_Hours_per_Night"
]

categorical_cols = [
    "Gender",
    "Department",
    # "Grade",  # excluded!
    "Extracurricular_Activities",
    "Internet_Access_at_Home",
    "Parent_Education_Level",
    "Family_Income_Level"
]

# Check for missing columns
missing_cols = [col for col in features_to_use + ['Pass_Fail'] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# Separate features and target
X = df[features_to_use]
y = df['Pass_Fail']

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Save feature columns after encoding
joblib.dump(X_encoded.columns.tolist(), "trained_data/pass_fail_features.pkl")

# Save categorical columns for API
joblib.dump(categorical_cols, "trained_data/categorical_cols.pkl")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Build pipeline and train
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Save the model pipeline
joblib.dump(pipeline, "trained_data/pass_fail_model.pkl")

print("✅ Training complete. Pass_Fail created from Grade; model saved.")