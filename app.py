from flask import Flask
from flask import request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model_a = tf.keras.models.load_model("./models/model_a.pb")
model_v = tf.keras.models.load_model("./models/model_v.pb")
model_d = tf.keras.models.load_model("./models/model_d.pb")

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict",methods = ['GET'])
def predictEmotion():
    if request.method == 'GET':
        data = request.args.get('data','')
        data = data.split(',')
        data = np.array(data, dtype = np.float64).reshape(-1,68 )

        valence = model_v.predict(data)
        dominance = model_d.predict(data)
        arousal = model_a.predict(data)
        return str([valence[0],dominance[0],arousal[0]])
app.run()