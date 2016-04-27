'''
Created on Apr 26, 2016

@author: Paul Reiners
'''
import pandas as pd 
import numpy as np
from sklearn.cross_validation import train_test_split
from util import log_loss 

def get_and_clean_data():
    dtype = {'Name':str}
    data = pd.read_csv('../data/train.csv', dtype=dtype, parse_dates=['DateTime'], index_col=0)
    categorical_columns = ["OutcomeType", "OutcomeSubtype", "AnimalType", "SexuponOutcome", "Breed", "Color"]
    for categorical_column in categorical_columns:
        data[categorical_column] = data[categorical_column].astype('category')
    
    data['AgeuponOutcome'] = data['AgeuponOutcome'].apply(convert_age_to_days)
    return data


def convert_age_to_days(age_str):
    if type(age_str) is str:
        parts = age_str.split()
        num = int(parts[0])
        unit = parts[1]
        if 'day' in unit:
            return num
        elif 'week' in unit:
            return 7 * num
        elif 'month' in unit:
            return 30 * num 
        elif 'year' in unit:
            return 365 * num 
    else:
        return np.nan
    
def get_black_dogs(data):
    black = data['AnimalType'] == "Dog"
    dog = data['Color'] == "Black"
    black_dogs = data[black & dog]

    return black_dogs

if __name__ == '__main__':
    data = get_and_clean_data()
    print data.dtypes
    X = data.drop(['OutcomeType'], axis=1)
    y = data['OutcomeType']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    predictions = [{'Adoption': 1, 'Died': 0, 'Euthanasia': 0, 'Return_to_owner': 0, 'Transfer': 0}] * len(y_test)
    possible_outcomes = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    ll = log_loss(y_test, predictions, possible_outcomes)
    print ll
