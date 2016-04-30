'''
Created on Apr 26, 2016

@author: Paul Reiners
'''
from math import log

from sklearn.cross_validation import train_test_split

import numpy as np
import pandas as pd


def log_loss(truths, label_col_name, predictions_df, possible_labels):
    ''' Function to measure log loss of a prediction.

    Parameters
    ==========
    truths          : numpy.ndarray
                      the ground truth
    label_col_name  : str
                      the name of the column you're trying to predict
    predictions_df  : pandas.core.frame.DataFrame
                      your predictions
    possible_labels : list
                      possible labels'''
    n = len(truths)
    total = 0.0
    for i in range(n):
        truth = truths[i]
        for possible_label in possible_labels:
            if truth == possible_label:
                y = 1
            else:
                y = 0
            prediction = predictions_df.iloc[i]
            p = prediction[truth]
            p = max(min(p, 1 - 1e-15), 1e-15)
            total += y * log(p)

    return -1.0 / n * total


def get_data(file_path, tag=None):
    dtype = {'Name': str}
    data = pd.read_csv(
        file_path, dtype=dtype, parse_dates=['DateTime'], index_col=0)
    data.Name = data.Name.fillna('')
    data.SexuponOutcome = data.SexuponOutcome.fillna('')
    if tag:
        data['tag'] = tag

    return data


def split_data(train_data):
    X = train_data.drop(['OutcomeType'], axis=1)
    y = train_data['OutcomeType']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, y_train, X_test, y_test


def get_is_named(name):
    if len(name) > 0:
        return 1.0
    else:
        return 0.0


def get_month(date_time):
    return date_time.month


def is_dog(animal_type):
    if animal_type == 'Dog':
        return 1.0
    else:
        return 0.0


def is_intact(sex_upon_outcome):
    if 'Intact' in sex_upon_outcome:
        return 1.0
    elif 'Neutered' in sex_upon_outcome or 'Spayed' in sex_upon_outcome:
        return 0.0
    else:
        return 0.5


def is_black(color):
    if color == 'Black':
        return 1.0
    else:
        return 0.0


def is_male(sex_upon_outcome):
    if 'Male' in sex_upon_outcome:
        return 1.0
    elif 'Female' in sex_upon_outcome:
        return 0.0
    else:
        return 0.5


def preprocess_data(data):
    data['Name'] = data['Name'].apply(get_is_named)
    data['Month'] = data['DateTime'].apply(get_month)

    if 'AnimalType' in data.columns:
        data['IsDog'] = data['AnimalType'].apply(is_dog)
        data = data.drop(['AnimalType'], axis=1)

    data['IsIntact'] = data['SexuponOutcome'].apply(is_intact)
    data['IsMale'] = data['SexuponOutcome'].apply(is_male)
    data['IsBlack'] = data['Color'].apply(is_black)

    drop_cols = ['OutcomeSubtype', 'DateTime', 'SexuponOutcome']
    data = data.drop(drop_cols, axis=1)
    categorical_columns = ["OutcomeType", "Breed", 'Month',
                           "Color"]
    for categorical_column in categorical_columns:
        data[categorical_column] = data[categorical_column].astype('category')
    data['AgeuponOutcome'] = data['AgeuponOutcome'].apply(convert_age_to_days)

    data = pd.get_dummies(data, columns=["Breed", 'Month',
                                         "Color"])

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
