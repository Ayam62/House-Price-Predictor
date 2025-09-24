from flask import  Flask,request,jsonify
import json
import pickle
import numpy as np
from flask_cors import CORS


app=Flask(__name__)
CORS(app)

with open ("artifacts/banglore_home_prices_model.pickle","rb") as f:
    model = pickle.load(f)

with open("artifacts/columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]


locations = data_columns[3:]
@app.route("/get_location_names")
def get_location_names():
    return jsonify({
        "locations": locations
    })

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index=-1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
        x = x.reshape(1, -1)
    return round(float(model.predict([x][0])),2)


@app.route("/predict_home_price", methods=["POST"])
def predict_home_price():
    total_sqft = float(request.form["total_sqft"])
    location = request.form["location"]
    bhk = int(request.form["bhk"])
    bath = int(request.form["bath"])

    response = jsonify({
        "estimated_price": get_estimated_price(location, total_sqft, bhk, bath)
    })

    response.headers.add("Access-Control-Allow-Origin","*")
    return response

if __name__=="__main__":
    print("Starting python flask server for House price prediction...")
    print(get_estimated_price("1st Phase JP Nagar", 1000, 2, 2))

    app.run()
