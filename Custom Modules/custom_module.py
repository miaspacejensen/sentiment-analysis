import zipfile
import pandas as pd
from pathlib import Path
import os

def collect_zip(dir, folder, zip_folder, file_name):
    '''
    Reads a file from a zipped folder, then converts it to a dataframe.

    Output is a dataframe.

    Parameter examples:

        folder = "Data" #name of root folder
        zip_folder = "Amazon Fine Food Reviews" #name of zipped folder
        file_name = "Reviews.csv" #name of file inside zipped folder
    '''

    os.chdir(dir)

    with zipfile.ZipFile(dir + "/" + folder + "/" + zip_folder + '.zip', 'r') as zip:
        file_list = zip.namelist()
        for file in file_list:
            if file.endswith(file_name):
                with zip.open(file) as f:
                    df = pd.read_csv(f)

    return df

def chunk(zip_dir, dest_dir, folder, zip_folder, file_name, chunks):
    '''
    Takes a large file and splits into chunks.

    Output is a series of CSV files.
    '''
    data = collect_zip(zip_dir, folder, zip_folder, file_name)

    data_size = len(data)
    chunk_start = 0
    chunk_size = chunk_end = round(len(data) / chunks)

    print("Data size:", data_size)
    print("Chunks:", chunks)
    print("Chunk size:", chunk_size)
    print("Starting range:", chunk_start, ",", chunk_end)

    print("Chunking...")
    for i in range(1, chunks+1):
        if i != chunks:
            df = data.iloc[chunk_start:chunk_end, :]
            chunk_start = chunk_end
            chunk_end += chunk_size
        if i == chunks:
            df = data.iloc[chunk_start:, :]
        print(df.shape)
        df.to_csv(f"{dest_dir}/{folder}/{zip_folder}/Part{i}_{file_name}")