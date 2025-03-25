# Importing required libraries
import pandas as pd               # For data manipulation
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For visualization
import seaborn as sns             # For advanced visualizations
import pickle                     # For saving the model

# Importing dataset
sample = pd.read_csv("dataset/dataset.csv")

# Displaying basic information about dataset
print(sample.head())  # First 5 rows
print(sample.tail())  # Last 5 rows
print(sample.columns) # Column names
print(sample.info())  # Summary of dataset
print(sample.describe())  # Statistical summary

# Checking missing values
print(sample.isnull().sum())  

# Handling categorical variables using Label Encoding
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

categorical_cols = [
    "Reached_On_Time", "Patient_Gender", "Test_Name", "Sample",
    "Way_Of_Storage_Of_Sample", "Cut_off_Schedule", "Traffic_Conditions", "Mode_Of_Transport"
]

for col in categorical_cols:
    sample[col] = lb.fit_transform(sample[col])

# Dropping unnecessary columns
drop_cols = ['Patient_ID', 'Patient_Age', 'Test_Booking_Date', 
             'Sample_Collection_Date', 'Agent_ID', 'Mode_Of_Transport']

sample1 = sample.drop(columns=drop_cols)

# Visualizing relationships
plt.figure(figsize=(10, 6))
plt.scatter(sample1["Time_Taken_To_Reach_Lab_MM"], sample1["Reached_On_Time"], color='blue')
plt.xlabel("Time Taken To Reach Lab (Minutes)")
plt.ylabel("Reached On Time")
plt.title("Time Taken vs Reached On Time")
plt.show()

sns.pairplot(sample1)
sns.jointplot(x=sample1["Time_Taken_To_Reach_Lab_MM"], y=sample1["Reached_On_Time"], kind="scatter")

# Splitting the dataset into Input (X) and Output (Y)
X = sample1.drop(columns=["Reached_On_Time"])  # Features
y = sample1["Reached_On_Time"]                 # Target variable

# Splitting into Training and Testing sets (75% training, 25% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)

# Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# Random Forest Model (Better than Decision Tree)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, criterion='gini', random_state=42)
rf_model.fit(X_train, y_train)

# Evaluating Random Forest Model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Saving the best model (Random Forest)
with open('model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Loading and testing the saved model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Testing the model on new data
test_sample = pd.DataFrame(X.iloc[0:1])  # Taking first row as test input
prediction = loaded_model.predict(test_sample)
print("Model Prediction:", "Sample is on time" if prediction[0] == 1 else "Sample is late")
