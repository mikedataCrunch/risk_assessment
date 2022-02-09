import requests
import os
import json
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(os.getcwd(), config['output_model_path'])

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1"

#Call each API endpoint and store the responses
response1 = requests.get(f'{URL}:8000/prediction?filedirectory=testdata').text
response2 =  requests.get(f'{URL}:8000/scoring').text
response3 =  requests.get(f'{URL}:8000/summarystats').text
response4 =  requests.get(f'{URL}:8000/diagnostics').text

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace, apireturns.txt
apireturns = "\n".join(responses)
fp = os.path.join(output_model_path, 'apireturns.txt')
with open(fp, 'w') as handle:
    handle.write(apireturns)

