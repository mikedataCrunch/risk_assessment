from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# create auxiliary function for reading and concatenating files in a dir
def readcsvs(source_dir):
    filenames = [file for file in os.listdir(source_dir) if file.endswith('.csv')]
    df_list = []
    for file in filenames:
        temp_df = pd.read_csv(os.path.join(source_dir, file))
        df_list.append(temp_df)
    return pd.concat(df_list)


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(os.getcwd(), config['output_model_path']) 
test_data_path = os.path.join(os.getcwd(), config['test_data_path']) 

#################Function for model scoring
def score_model(test_data_path, model_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    test_df = readcsvs(test_data_path)
    
    X_test = test_df.drop(columns=['corporation', 'exited'])
    y_test = test_df['exited']

    # load model
    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb') as handle:
        model = pickle.load(handle)

    y_pred = model.predict(X_test)
    score = metrics.f1_score(y_test, y_pred)
    scorelog = open('latestscore.txt', 'w')
    scorelog.write(str(score))
    scorelog.close()

if __name__ == "__main__":
    score_model(test_data_path, model_path)
