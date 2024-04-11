from flask import Flask, render_template, request
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('ml_project\employees.csv')

# Drop unnecessary columns
data = data[['EMPLOYEE_ID', 'SALARY', 'POSITION', 'YEARS_EXPERIENCE']]

# Drop rows with missing values
data.dropna(inplace=True)

# Splitting the dataset into features and target variable
X = data[['SALARY', 'YEARS_EXPERIENCE']]
y = data['POSITION']

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the KNN model
k = 5  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)


# Function to predict position and generate plot
def predict_position(salary, years_experience):
    # Prepare the input data
    input_data = scaler.transform([[salary, years_experience]])
    
    # Predict position
    position = knn.predict(input_data)[0]
    
    # Generate plot
    plt.scatter(data['SALARY'], data['YEARS_EXPERIENCE'], color='blue', label='Actual')
    plt.scatter(salary, years_experience, color='red', label='Predicted')
    plt.title('Actual vs Predicted Position')
    plt.xlabel('Salary')
    plt.ylabel('Years of Experience')
    plt.legend()
    
    # Convert plot to base64 encoded image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    
    return position, plot_data


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        years_experience = int(request.form['years_experience'])
        salary = int(request.form['salary'])
        
        # Predict position based on years of experience
        predicted_position, plot_data = predict_position(salary, years_experience)
        
        return render_template('result.html', years_experience=years_experience, salary=salary,
                               predicted_position=predicted_position, plot_data=plot_data)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
