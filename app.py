import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from flask import Flask, jsonify, request
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model import load_model

app = Flask(__name__)
start_time = time.time()


@app.route('/')
def hello_world():  # put application's code here
    return {"message": "hello world"}


fille_name: str = "model.pkl"


@app.route('/predict', methods=['GET'])
def predict():
    try:
        if not request.is_json:
            raise Exception("Request must be json")
        model: LogisticRegression = load_model(fille_name)
        data = request.get_json()
        iris = load_iris()
        labels = iris.target_names
        try:
            X = np.array(
                [data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]).reshape(1, -1)
        except KeyError as e:
            return jsonify({"KeyError": f"No feature: {e}"}), 400
        predicted_class = labels[model.predict(X)[0]]
        return jsonify({"label-index": int(model.predict(X)[0]), "label-prediction": predicted_class}), 200
    except Exception as e:
        return jsonify({"Error": f"{e}"}), 400


@app.route('/info', methods=['GET'])
def info():
    model: LogisticRegression = load_model("model.pkl")
    model_info: dict = {
        "model-type": str(type(model)),
        "model-params": model.get_params(),
        "model-features": str(model.coef_.shape[1]),
        "model-bias": str(model.intercept_)
    }
    return jsonify(model_info), 200


@app.route('/health', methods=['GET'])
def health():
    server_info: dict = {
        "server-info": {
            "status": "ok",
            "uptime": f"{time.time() - start_time:.2f} seconds"
        }
    }
    return jsonify(server_info), 200


@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    model = load_model(fille_name)
    X, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return jsonify(accuracy_score(y_test, model.predict(x_test)))


if __name__ == '__main__':
    app.run()
