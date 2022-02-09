
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(os.getcwd(), config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(), config['test_data_path']) 

# create auxiliary function for reading and concatenating files in a dir
def readcsvs(source_dir):
    filenames = [file for file in os.listdir(source_dir) if file.endswith('.csv')]
    df_list = []
    for file in filenames:
        temp_df = pd.read_csv(os.path.join(source_dir, file))
        df_list.append(temp_df)
    return pd.concat(df_list)
data = readcsvs(test_data_path)
def load_model(path):
    with open(path, 'rb') as handle:
        model = pickle.load(handle)
    return model

##################Function to get model predictions
def model_predictions(data):
    #read the deployed model and a test dataset, calculate predictions
    dropcols = ['corporation']
    if 'exited' in data.columns:
        dropcols = dropcols + ['exited']
    X = data.drop(columns=dropcols)
    model_path = os.path.join(
        os.getcwd(), config['prod_deployment_path'],
        'trainedmodel.pkl'
    )
    model = load_model(model_path)
    y_preds = model.predict(X)

    #return value should be a list containing all predictions
    return list(y_preds.ravel())

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = readcsvs(output_folder_path)
    num_df = df.select_dtypes(include=np.number)
    mean_list = num_df.mean(axis=0).to_list()
    std_list = num_df.std(axis=0).to_list()
    median_list = num_df.median(axis=0).to_list()    
    return [mean_list, std_list, median_list]

def dataframe_nulls():
    # get null percentages
    df = readcsvs(output_folder_path)
    return (df.isna().sum() / len(df.index)).to_list()

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingest_duration = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    os.system('python training.py')
    train_duration = timeit.default_timer() - start_time
    return [ingest_duration, train_duration]

##################Function to check dependencies
def outdated_packages_list():
    #get a list of oudated packges
    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    
    with open('outdated.txt', 'wb') as f:
        f.write(outdated)

if __name__ == '__main__':
    model_predictions(data)
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    
