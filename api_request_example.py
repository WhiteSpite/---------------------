import requests
import json
import numpy as np


'''Server agent plays with O'''

X = 'X'
O = 'O'
_ = '_'

request_state = [[_, _, _], 
                 [_, X, _], 
                 [_, _, _]]

url = "http://45.130.43.227:5000/api"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, data=json.dumps(request_state), headers=headers)
    response_state = np.array(json.loads(response.text))
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(response_state)
except requests.RequestException as e:
    print(f"An error occurred: {e}")
