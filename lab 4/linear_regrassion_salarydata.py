import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('boston_housing.csv')

# Define the independent variables
X = data[['rm', 'crim', 'age']]
y = data['medv']

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
x = data['rm']
y = data['crim']
z = data['age']
ax.scatter(x, y, z, c='b', marker='o')

# Fit a plane using model coefficients
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 10), np.linspace(y.min(), y.max(), 10))
zz = model.coef_[0] * xx + model.coef_[1] * yy + model.coef_[2] * z.mean() + model.intercept_

# Plot the surface
ax.plot_surface(xx, yy, zz, alpha=0.5)

# Add labels and title
ax.set_xlabel('Average Number of Rooms')
ax.set_ylabel('Per Capita Crime Rate')
ax.set_zlabel('Proportion of Owner-Occupied Units Built Prior to 1940')
plt.title('Multiple Linear Regression')

# Show the plot
plt.show()
