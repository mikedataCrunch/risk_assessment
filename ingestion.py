import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    # read all files in input_folder
    source_dir = os.path.join(os.getcwd(), input_folder_path)
    filenames = [file for file in os.listdir(source_dir) if file.endswith('.csv')]
    df_list = []
    for file in filenames:
        temp_df = pd.read_csv(os.path.join(source_dir, file))
        df_list.append(temp_df)
    
    # combine
    df = pd.concat(df_list)
    
    # de-dup
    df = df.drop_duplicates()

    # save
    dest_dir = os.path.join(os.getcwd(), output_folder_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    save_fname = 'finaldata.csv'
    df.to_csv(os.path.join(dest_dir, save_fname), index=False)
    
    # record ingested files
    with open('ingestedfiles.txt', 'w') as f:
        f.write(str(filenames))



if __name__ == '__main__':
    merge_multiple_dataframe()
