from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path']) 
model = os.path.join(os.getcwd(), config['output_model_path'],'trainedmodel.pkl')

if not os.path.exists(prod_deployment_path):
    os.makedirs(prod_deployment_path)

####################function for deployment
def store_model_into_pickle(model, prod_deployment_path):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    os.system(f'cp {model} {prod_deployment_path}')
    os.system(f'cp latestscore.txt {prod_deployment_path}')
    os.system(f'cp ingestedfiles.txt {prod_deployment_path}')

if __name__=="__main__":
    store_model_into_pickle(model, prod_deployment_path)

