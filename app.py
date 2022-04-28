from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import requests
app = Flask(__name__)
CORS(app)

@app.route("/mood")
#Get the mood with the help of device
def mood():
    data = requests.get("http://192.168.43.86").content
    data = data.split()
    data = np.array(data, dtype=np.float64).reshape(-1, 42)
    model_a = tf.keras.models.load_model('saved_model/modev_a')
    model_v = tf.keras.models.load_model('saved_model/modev_v')
    model_d = tf.keras.models.load_model('saved_model/modev_d')
    valence = model_v.predict(data)
    dominance = model_d.predict(data)
    arousal = model_a.predict(data)
    response = jsonify(
        str([valence[0][0], dominance[0][0], arousal[0][0]]))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

app.run()
