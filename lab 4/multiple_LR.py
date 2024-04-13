import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('50_Startups.csv')

# Define the independent and dependent variables
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Model evaluation
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error: %.2f' % mean_squared_error(y, y_pred))
print('Coefficient of determination (R^2): %.2f' % r2_score(y, y_pred))

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add the data points
x = data['R&D Spend']
y = data['Administration']
z = data['Marketing Spend']
ax.scatter(x, y, z, c=y_pred, cmap='viridis')

# Add labels and title
ax.set_xlabel('R&D Spend')
ax.set_ylabel('Administration')
ax.set_zlabel('Marketing Spend')
plt.title('Multiple Linear Regression')

# Show the plot
plt.show()
