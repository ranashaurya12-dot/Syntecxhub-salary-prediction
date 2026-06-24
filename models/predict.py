import joblib
import pandas as pd

# Load model
model = joblib.load("models/best_model.pkl")

# Input
experience = 4

# Create DataFrame with only the feature the model expects
data = pd.DataFrame({
    "experience": [experience]
})

# Predict
prediction = model.predict(data)

print("Predicted Salary =", round(prediction[0], 2))