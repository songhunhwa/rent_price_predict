import numpy as np
import pickle
import json

# load model & scaler
model = pickle.load(open('finalized_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# sample json
with open('example_json.json') as json_file:
    example_json = json.load(json_file)       

# predict
example_data = np.fromiter(example_json.values(), dtype=float).reshape(1, -1)
data_scaled = scaler.transform(example_data)

print(model.predict(data_scaled))