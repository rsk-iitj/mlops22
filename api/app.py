from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    model_name = request.json['mode-name']
    print("done loading")
    model_path = "../models/"+model_name+".joblib"
    model = load(model_path)

    predicted = model.predict([image])
    return {"y_predicted": int(predicted[0])}

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')
    