'''
Created on May 2, 2016

@author: Paul Reiners
'''
import pandas as pd


def get_data(file_path, tag=None):
    dtype = {'Name': str}
    data = pd.read_csv(
        file_path, dtype=dtype, parse_dates=['DateTime'], index_col=0)
    data.Name = data.Name.fillna('')
    data.SexuponOutcome = data.SexuponOutcome.fillna('')
    if tag:
        data['tag'] = tag

    return data
