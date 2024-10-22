from flask import Flask, request , jsonify
import datetime
import pickle
import numpy as np
# загружаем модель из файла
with open('./model/model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)

# создаём приложение
app = Flask(__name__)

@app.route('/')
def index():
    msg = "Test message. The server is running"
    return msg


@app.route('/predict', methods=['POST'])
def predict():

    features = request.json
    columns = ['status', 'baths', 'city', 'sqft', 'zipcode', 'state', 'pool', 'Type', 'Yearbuilt', 'Heating', 'Cooling', 'Parking', 'fireplace', 'school_ratimg _mean', 'school_distance_meann', 'lat', 'lng']
	date_model = pd.DataFrame([features], columns=columns)
	
	for column in ['baths', 'sqft', 'school_ratimg _mean', 'school_distance_mean', 'lat', 'lng']:
		date_model[column] = date_model[column].apply(lambda x: abs(x))
		cons = 1e-1
		date_model[column] = np.log(date_model[column] + cons)

    pred = model.predict(date_model)
    return jsonify({'prediction': round(np.exp(pred[0]))})
   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
