from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(os.getcwd(), config['output_folder_path']) 
model_path = os.path.join(os.getcwd(), config['output_model_path']) 

# create auxiliary function for reading and concatenating files in a dir
def readcsvs(source_dir):
    filenames = [file for file in os.listdir(source_dir) if file.endswith('.csv')]
    df_list = []
    for file in filenames:
        temp_df = pd.read_csv(os.path.join(source_dir, file))
        df_list.append(temp_df)
    return pd.concat(df_list)


#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    df = readcsvs(dataset_csv_path)

    X = df.drop(columns=['corporation', 'exited'])
    y = df['exited']
    model.fit(X, y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    save_fp = os.path.join(model_path, 'trainedmodel.pkl')
    with open(save_fp, 'wb') as handle:
        pickle.dump(model, handle)

if __name__ == "__main__":
    train_model()