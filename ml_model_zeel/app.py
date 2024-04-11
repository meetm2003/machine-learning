from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load the pre-trained model
model = LogisticRegression(max_iter=1000)
data = pd.read_csv("telecust.csv")
X = data[['income', 'age', 'gender']]  # Features: income, age, gender
y = data['custcat']  # Target variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)

# Define a route for prediction and graph display
@app.route('/', methods=['GET', 'POST'])
def predict_and_display_graph():
    if request.method == 'POST':
        # Get the data from the form
        income = float(request.form['income'])
        age = float(request.form['age'])
        gender = float(request.form['gender'])
        
        # Preprocess the features
        features = scaler.transform([[income, age, gender]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Plot the distribution of predicted custcat values
        plt.hist(y, bins=range(1, 6), alpha=0.5, label='Actual custcat')
        plt.axvline(x=prediction, color='red', linestyle='dashed', linewidth=1, label='Predicted custcat')
        plt.xlabel('custcat')
        plt.ylabel('Frequency')
        plt.title('Distribution of custcat')
        plt.legend()
        
        # Ensure the static directory exists
        static_dir = os.path.join(app.root_path, 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        
        # Save the graph image in the static directory
        plt.savefig(os.path.join(static_dir, 'prediction_graph.png'))
        plt.close()
        
        # Return the prediction and render the template with the graph
        return render_template('index.html', prediction=prediction, image_path='static/prediction_graph.png')
    
    # If it's a GET request, just render the template
    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
