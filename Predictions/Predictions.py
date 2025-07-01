import joblib
import pandas as pd
model = joblib.load(r'G:\My Drive\AI\Machine Learning\Supervised Learing\Projects\CardioRegression\Training Model\regression_model.pkl')

# Example: new data point to predict CaloriesBurned
new_data = pd.DataFrame([[40, 130, 30, 75]], columns=['Duration','HeartRate','Age','Weight'])

# Predict
predicted_calories = model.predict(new_data)

print("Predicted Calories Burned:", predicted_calories)