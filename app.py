import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def cancerPrediction():
    # Load data
    df = pd.read_csv("breast_cancer_data.csv")

    # Get input values and convert them to float
    try:
        inputQuery1 = float(request.form['query1'])
        inputQuery2 = float(request.form['query2'])
        inputQuery3 = float(request.form['query3'])
        inputQuery4 = float(request.form['query4'])
        inputQuery5 = float(request.form['query5'])
    except ValueError:
        return render_template('home.html', output1="Invalid input. Please enter numeric values.", query1="", query2="", query3="", query4="", query5="")

    # Features
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean']
    X = df[features]
    y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convert diagnosis to binary

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    # Model
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    model.fit(X_train, y_train)

    # Prediction
    input_data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
    new_df = pd.DataFrame(input_data, columns=features)

    prediction = model.predict(new_df)[0]
    proba = model.predict_proba(new_df)[0][1]  # probability of class "1" (cancer)

    if prediction == 1:
        output1 = "⚠️ The patient is diagnosed with Breast Cancer."
        output2 = "Confidence: {:.2f}%".format(proba * 100)
    else:
        output1 = "✅ The patient is not diagnosed with Breast Cancer."
        output2 = "Confidence: {:.2f}%".format((1 - proba) * 100)

    # Return values to template
    return render_template(
        'home.html',
        output1=output1,
        output2=output2,
        query1=inputQuery1,
        query2=inputQuery2,
        query3=inputQuery3,
        query4=inputQuery4,
        query5=inputQuery5
    )


    if __name__ == '__main__':
    app.run(debug=True)