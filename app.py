# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np


# app = Flask(__name__)


# with open('ada_model.pkl', 'rb') as file:
#     model = pickle.load(file)


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
    
#     data = request.json

    
#     rank = float(data['rank'])
#     percentile = float(data['percentile'])
#     gender = int(data['gender'])
#     category = int(data['category'])

    
#     features = np.array([[rank, percentile, gender, category]])

    
#     prediction = model.predict(features)


#     return jsonify({'prediction': int(prediction[0])})


# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and the label encoder
with open('ada_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from JSON
        rank = data.get('rank', None)
        percentile = data.get('percentile', None)
        gender = data.get('gender', None)
        category = data.get('category', None)

        # Validate inputs
        if rank is None or percentile is None or gender not in [0, 1] or category not in [0, 1, 2, 3]:
            return jsonify({'error': 'Invalid input. Please check your values.'}), 400

        # Prepare the input for the model
        input_data = pd.DataFrame([[rank, percentile, gender, category]],
                                   columns=['rank', 'percentile', 'gender', 'category'])

        # Make prediction
        prediction_encoded = model.predict(input_data)
        
        # Decode the prediction back to the original college name
        prediction_decoded = label_encoder.inverse_transform(prediction_encoded)

        # Return the prediction as JSON
        return jsonify({'predicted_college': prediction_decoded[0]})

    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


