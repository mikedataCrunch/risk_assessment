import training
import scoring
import deployment
import diagnostics
import reporting

import pandas as pd
import ast
import os
import json

with open('config.json','r') as f:
    config = json.load(f) 
##################Check and read new data
#first, read ingestedfiles.txt
prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path'])
ingested_fp = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
with open(ingested_fp, 'r') as f:
    ingested_files = ast.literal_eval(f.read())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_folder_path = os.path.join(os.getcwd(), config['input_folder_path'])
# runs ingestion.py if atleast 1 csv file is not in 'ingestedfiles.txt'
new_data_flag = False
new_files = []
for file in os.listdir(input_folder_path):
    if file not in ingested_files:
        # confirm if this is a datafile
        if file.endswith('.csv'):
            new_files.append(file)
            new_data_flag = True

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
test_data_path = os.path.join(os.getcwd(), config['test_data_path']) 
if new_data_flag:
    os.system('python ingestion.py')

    # get new dataset from new files
    new_data_list = []
    for file in new_files:
        temp_df = pd.read_csv(os.path.join(input_folder_path, file))
        new_data_list.append(temp_df)
    test_data = pd.concat(new_data_list)

    # replace the previous test data with a new one, the new files
    test_data_fname = os.path.join(test_data_path, 'testdata.csv')
    test_data.to_csv(test_data_fname, index=False)

else:
    print("No new files, end process.")
    exit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
latestscore_fp = os.path.join(prod_deployment_path, 'latestscore.txt')
with open(latestscore_fp, 'r') as f:
    latest_score = float(f.read())

model_drift_flag = False
# read new test data, and score old model on it
scoring.score_model(test_data_path, model_path=prod_deployment_path)

# read new score for comparison
with open('latestscore.txt', 'r') as f:
    new_score = float(f.read())
if new_score < latest_score:
    model_drift_flag = True

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here

##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
if model_drift_flag:
    # save new trained model in /models
    os.system('python training.py')

    # get new score after training
    # this updates the latestscore.txt in prod_deployment_path
    # assumes new score after retraining is better than old score
    # no more check employed before replacing old score in prod
    os.system('python scoring.py') 

    # deploy trainedmodel.pkl, along with ingestedfiles.txt, and latestscore.txt
    os.system('python deployment.py')
else:
    print("No model drift, end process.")
    exit()

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python reporting.py')
os.system('python apicalls.py')

print("Process Done!")




