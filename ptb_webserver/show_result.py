import json

from flask import Flask, jsonify, request
from flask.helpers import send_from_directory
from flask.templating import render_template

import os
import pickle
import numpy as np

import result_serving


app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("hello.html")


@app.route('/predict')
def predict():
    raw_input = request.args.get('path')
    raw_input = str(raw_input)

    top_k = request.args.get('topK')
    if top_k is None:
        top_k = 3

    softmax_output, path_representation, _ = result_serving.inference(raw_input)
    input_file = open("static/vocabulary/id_to_url.pkl", 'rb')
    id_to_url = pickle.load(input_file)
    input_file.close()

    prediction_id = list()
    tmp = softmax_output
    for i in range(top_k):
        max_id = np.argmax(tmp)
        prediction_id.append(max_id)
        tmp[max_id] = 0

    prediction_url = [id_to_url[index] for index in prediction_id]

    path = raw_input.split(",")
    data = {"path": path, "prediction_url": prediction_url, "path_representation": path_representation}

    return render_template("hello.html", data=data)


if __name__ == '__main__':
    app.run()
