import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def make_prediction(input_data: dict, model_path: str):
    """
    Takes just ONE passenger's data and predict that "Is the passenger survived or not?"
    """

    # 1. Load the Model (Pipeline)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model could not be found! You should run training_pipeline.py first.")

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # 2. Turn the data has come into DataFrame
    df = pd.DataFrame([input_data])

    print(f"Data has come: {input_data}")

    # 3. Make Prediction
    prediction = pipeline.predict(df)
    probability = pipeline.predict_proba(df)

    result = "Survived" if prediction[0] == 1 else "Didn't Survive"
    confidence = probability[0][prediction[0]]

    print(f"Result: {result} (Probability: {confidence * 100:.2f})")
    return result

if __name__ == "__main__":
    jack_dawson = {
        'PassengerId': 999,
        'Name': 'Jack Dawson',
        'Pclass': 3,
        'Sex': 'male',
        'Age': 20,
        'SibSp': 0,
        'Parch': 0,
        'Ticket': 'A/5 21171',
        'Fare': 7.25,
        'Cabin': np.nan,
        'Embarked': 'S'
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "titanic_pipeline.pkl")

    try:
        make_prediction(jack_dawson, model_path)
    except Exception as e:
        print(f"ERROR: {e}")