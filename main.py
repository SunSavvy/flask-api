from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import requests
import os

# Create a Flask app
app = Flask(__name__)

# default endpoint


@app.route("/", methods=["GET"])
def defaultRoute():
    return jsonify({
        'status': 'success',
        'message': 'Flask API ready'
    }), 200

# Define the route for making predictions


@app.route('/predict', methods=['POST'])
def predict():
    # get request body data
    reqBody = request.get_json()
    # print(reqBody)

    cityName = reqBody['city'].replace(" ", "")
    cityName = cityName.lower()
    # print(cityName)

    data = reqBody['prevUv']
    # print(data)
    reshaped_data = np.array(data).reshape(1, 24, 1)

    # get the right model url
    modelUrl = requests.get(
        f"https://storage.googleapis.com/sunsavvy/Model/{cityName.upper()}.h5")

    # create local directory to save the downloaded model
    modelDirectory = "models"
    if not os.path.exists(modelDirectory):
        os.makedirs(modelDirectory)

    # create local model template name
    modelFile = os.path.join(modelDirectory, f"{cityName.upper()}.h5")
    print(modelFile)

    with open(modelFile, 'wb') as f:
        f.write(modelUrl.content)

    # load model
    model = tf.keras.models.load_model(modelFile)

    # Convert the reshaped data to a list
    reshaped_data_list = reshaped_data.tolist()
    predictions = model.predict(reshaped_data_list)

    return jsonify({'status': 'success', 'message': 'Prediction result is out', 'predictions': predictions.tolist()}), 200


# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port)
