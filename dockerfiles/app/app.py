import json
from flask import Flask, jsonify, request
import mlflow
import pandas as pd
from pandas.tseries.offsets import DateOffset


app = Flask(__name__)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    month = json.loads(request.args["month"])
    offset = json.loads(request.args["offset"])
    df = pd.DataFrame({
    'ds': pd.date_range(
        start = pd.Timestamp(month),                        
        end = pd.Timestamp(month) + DateOffset(offset), 
        freq = 'D'
         )
    })
    model_uri = "./prophet"
    model = mlflow.pyfunc.load_model(model_uri)
    forecast = model.predict(df)
    result = forecast.to_json(orient='records')
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

