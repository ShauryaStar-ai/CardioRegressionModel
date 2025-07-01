# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# Load the data
dataset = pd.read_excel('trainnigSet.xlsx')
# Declare features and target
X = dataset[['Duration', 'HeartRate', 'Age', 'Weight']]
y = dataset['CaloriesBurned']
# Split the data: 80% train(that we will use for traning here), 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Save feature names for plotting
X_features = X.columns
# Initialize scaler
scaler = StandardScaler()
# Fit scaler on training data and transform training data
X_train_scaled = scaler.fit_transform(X_train)
# Transform test data using the same scaler (do NOT fit again)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Plot features vs target on training data (unscaled)
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
"""for i in range(4):
    ax[i].scatter(X_train.iloc[:, i], y_train, alpha=0.7)
    ax[i].set_xlabel(X_features[i])
    ax[i].set_title(f"{X_features[i]} vs CaloriesBurned")
ax[0].set_ylabel("CaloriesBurned")
plt.tight_layout()
plt.show()"""

# Plot features vs target on training data (scaled)

"""for i in range(4):
    ax[i].scatter(X_train_scaled_df.iloc[:, i], y_train, alpha=0.7)
    ax[i].set_xlabel(X_features[i])
    ax[i].set_title(f"{X_features[i]} vs CaloriesBurned")
ax[0].set_ylabel("CaloriesBurned")
plt.tight_layout()
plt.show()"""


# Train the Linear Regression model on scaled training data
model = LinearRegression()
model.fit(X_train_scaled_df, y_train)

# Predict on test data (scaled)
y_pred = model.predict(X_test_scaled)

# Evaluate model on test set

# Optional: save the trained model for later use
joblib.dump(model, 'regression_model.pkl')
