import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("D:\docs\Study-Materials-BVM-22CP308\sem 6\python ML\machine-learning\lab 5\IRIS.csv")
df.columns = ["X1", "X2", "X3", "X4", "Y"]

# Prepare the data
X = df.drop(columns=['Y'])
y = df['Y']

# Convert class labels to numerical values using label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

trainX, testX, trainY, testY = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train the decision tree classifier
decision = DecisionTreeClassifier(criterion="gini")
decision.fit(trainX, trainY) 

# Compute accuracy
accuracy = decision.score(testX, testY)
print("Accuracy:", accuracy)

# Visualize the decision tree
plt.figure(figsize=(10, 8))
plot_tree(decision, feature_names=list(X.columns), class_names=label_encoder.classes_, filled=True)
plt.show()
