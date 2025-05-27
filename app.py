from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('trained_data/pass_fail_model.pkl')
features = joblib.load('trained_data/pass_fail_features.pkl')
categorical_cols = joblib.load('trained_data/categorical_cols.pkl')

@app.route('/')
def home():
    return "Student Pass/Fail Predictor API is running!"

@app.route('/student-eval', methods=['POST'])
def student_eval():
    data = request.json

    try:
        input_df = pd.DataFrame([data])

        # One-hot encode only the categorical columns present in input
        input_cat = pd.get_dummies(input_df[categorical_cols])

        # Drop original categorical columns and concat encoded ones
        input_df = input_df.drop(columns=categorical_cols)
        input_df = pd.concat([input_df, input_cat], axis=1)

        # Add any missing columns from training set as zero
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training data exactly
        input_df = input_df[features]

        prediction = model.predict(input_df)[0]
        status = "Pass" if prediction == 1 else "Fail"

        return jsonify({"prediction": status})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
