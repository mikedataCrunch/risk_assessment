from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle

import json
import os

import diagnostics 
import scoring



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(os.getcwd(), config['output_folder_path']) 
prod_model_path = os.path.join(os.getcwd(), config['prod_deployment_path'])

def readcsvs(source_dir):
    filenames = [file for file in os.listdir(source_dir) if file.endswith('.csv')]
    df_list = []
    for file in filenames:
        temp_df = pd.read_csv(os.path.join(source_dir, file))
        df_list.append(temp_df)
    return pd.concat(df_list)

#######################Prediction Endpoint
@app.route("/prediction")
def predict():        
    #call the prediction function you created in Step 3
    filedirectory = request.args.get('filedirectory')
    if filedirectory:
        data = readcsvs(filedirectory)
        y_pred = diagnostics.model_predictions(data)
        return "\nPredictions: " + str(y_pred) + "\n\n" #add return value for prediction outputs
    else:
        return "\nQuery invalid! Input a filedirectory query\n\n"
    

#######################Scoring Endpoint
@app.route("/scoring")
def score():        
    #check the score of the deployed model
    test_data_path = os.path.join(os.getcwd(), config['test_data_path'])
    model_path = prod_model_path
    scoring.score_model(test_data_path, model_path)
    with open('latestscore.txt', 'r') as f:
        score = f.read()
    return "Score: " + str(score)  + "\n" #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats")
def stats():        
    #check means, medians, and modes for each column
    #return a list of all calculated summary statistics
    summary_stats = diagnostics.dataframe_summary()
    return  f"Summary statistics\n" + \
    f"column means: {str(summary_stats[0])}\n" +\
    f"column standard dev: {str(summary_stats[1])}\n" +\
    f"column medians: {str(summary_stats[2])}\n"

#######################Diagnostics Endpoint
@app.route("/diagnostics")
def diagnosis():        
    #check timing and percent NA values
    null_count_list = diagnostics.dataframe_nulls()
    execution_time_list = diagnostics.execution_time()
    diagnostics.outdated_packages_list()
    with open('outdated.txt', 'rb') as f:
        outdated_report = f.read()
    
    return f"\nNull counts: {str(null_count_list)}\n" + \
        f"Execution times in seconds [ingest, train]: \n{str(execution_time_list)}" +\
        f"\nDependencies Report:\n{str(outdated_report)}"

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
