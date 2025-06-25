# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_excel('trainnigSet.xlsx')
# the Dataset is there and now fully working

# input and then output variable
X = dataset[['Duration','HeartRate','Age','Weight']]
y = dataset['CaloriesBurned']
model = LinearRegression()
model.fit(X ,y)

# 5. Make predictions
y_pred = model.predict(X)

    #Showing the model and the point

# Example: new data point to predict CaloriesBurned
new_data = pd.DataFrame([[40, 130, 30, 75]], columns=['Duration','HeartRate','Age','Weight'])

# Predict
predicted_calories = model.predict(new_data)

print("Predicted Calories Burned:", predicted_calories)
