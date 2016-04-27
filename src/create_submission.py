'''
Created on Apr 26, 2016

Best predictors:
    Date        Type              MyLLScore  KaggleLLScore  GithubTag
    ====================================================================
    04/27/2016  BaseLinePredictor  20.61577       20.25113  Submission00

@author: Paul Reiners
'''
import pandas as pd 
import numpy as np
from sklearn.cross_validation import train_test_split
from util import log_loss 
from base_line_predictor import BaseLinePredictor

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

if __name__ == '__main__':
    data = get_and_clean_data()
    X = data.drop(['OutcomeType'], axis=1)
    y = data['OutcomeType']
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    predictor = BaseLinePredictor()
    predictions_df = predictor.predict(X_train, X_test, y_train)
    
    possible_outcomes = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    ll = log_loss(y_test, 'OutcomeType', predictions_df, possible_outcomes)
    print "%.5f" % ll
