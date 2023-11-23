import requests
import json
import numpy as np

X = 'X'
O = 'O'
_ = '_'

state = [[_, _, O], 
         [_, _, X], 
         [X, O, _]]

url = "http://45.130.43.227:5000/api"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(state), headers=headers)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    state = np.array(json.loads(response.text))
    print(state)
except requests.RequestException as e:
    print(f"An error occurred: {e}")
