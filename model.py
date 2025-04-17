# Importing required libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pickle  # Import pickle for saving the model

# Load the dataset
df = pd.read_csv('dataset_final.csv')

# Display data information and unique values
df.info()
print(df.nunique())

# Splitting dataset into features (X) and target (y)
X = df[['category', 'rank', 'percentile',  'gender']]
y = df['college_branch']

# Label encoding for categorical columns
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'category', 'college_branch']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Updating features (X) and target (y) after encoding
X = df[['rank', 'percentile', 'gender', 'category']]
y = df['college_branch']

# Displaying the head of the feature and target datasets
print("Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)



# Initializing and fitting AdaBoostClassifier
AdaBoostClf = AdaBoostClassifier(n_estimators=2500, learning_rate=0.1)
model = AdaBoostClf.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('ada_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved to ada_model.pkl")

# Predicting on the test set
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Calculating accuracy, error rate, precision, recall, F1-score, and R-squared (R^2) score
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)


decoded_college_names = label_encoder.inverse_transform(y_pred)

# Loading the model back from the pickle file (optional, for testing the loaded model)
with open('ada_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# You can use `loaded_model` to make predictions again to verify it works
new_predictions = loaded_model.predict(X_test)
print("\nPredictions from loaded model:")
print(new_predictions)


import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('dataset_final.csv')

# Splitting dataset into features (X) and target (y)
X = df[['category', 'rank', 'percentile',  'gender']]
y = df['college_branch']

# Label encoding for categorical columns
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'category']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Encoding the target variable (college_name)
y_encoded = label_encoder.fit_transform(y)

# Updating features (X) and target (y) after encoding
X = df[['rank', 'percentile', 'gender', 'category']]

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=100)

# Initializing and fitting AdaBoostClassifier
AdaBoostClf = AdaBoostClassifier(n_estimators=2500, learning_rate=0.1)
model = AdaBoostClf.fit(X_train, y_train)

# Save the trained model and the label encoder to pickle files
with open('ada_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

print("Model and label encoder saved.")

# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Load the model and label encoder
# with open('ada_model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# with open('label_encoder.pkl', 'rb') as encoder_file:
#     label_encoder = pickle.load(encoder_file)

# # Load your dataset to get college names
# df = pd.read_csv('DATA12.csv')
# college_names = df['college_name'].unique()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.form
#     rank = int(data['rank'])
#     percentile = float(data['percentile'])
#     gender = data['gender']
#     category = data['category']

#     # Prepare the input data for prediction
#     input_data = pd.DataFrame({
#         'rank': [rank],
#         'percentile': [percentile],
#         'gender': [gender],
#         'category': [category]
#     })

#     # Encode the categorical features
#     for col in ['gender', 'category']:
#         input_data[col] = label_encoder.transform(input_data[col])

#     # Predict college name
#     predictions = model.predict(input_data)
    
#     # Get the top 3 college names
#     top_colleges = label_encoder.inverse_transform(predictions)
#     top_colleges_list = list(top_colleges)
#     return jsonify(top_colleges_list[:3])  # Return the top 3 colleges

# if __name__ == '__main__':
#     app.run(debug=True)
