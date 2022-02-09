import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(os.getcwd(), config['output_model_path'])

def readcsvs(source_dir):
    filenames = [file for file in os.listdir(source_dir) if file.endswith('.csv')]
    df_list = []
    for file in filenames:
        temp_df = pd.read_csv(os.path.join(source_dir, file))
        df_list.append(temp_df)
    return pd.concat(df_list)

##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    data = readcsvs(test_data_path)
    y_pred = diagnostics.model_predictions(data)
    y_true = data['exited']
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))

if __name__ == '__main__':
    score_model()
