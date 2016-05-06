'''
Created on Apr 26, 2016

Functions for selecting best features.

@author: Paul Reiners
'''
from core.preprocessing.feature_extraction_scaling \
    import extract_features
import pandas as pd


def select_features(data, animal_type):
    data = data[data['AnimalType'] == animal_type]
    data = data.drop(['AnimalType'], axis=1)

    data = extract_features(data, animal_type)

    if 'tag' in data.columns:
        keep_cols = ['OutcomeType', 'tag']
    else:
        keep_cols = ['OutcomeType']
    if animal_type == 'Cat':
        keep_cols.extend(
            ['AgeuponOutcome', 'IsSpring', 'IsChristmas', 'IsJuly',
             'IsWeekend', 'IsMidnight', 'IsEightAM', 'IsNineAM', 'IsTenAM',
             'IsFivePM', 'IsSixPM', 'IsThreePM', 'IsNamed', 'IsIntact'])
    else:
        keep_cols.extend(
            ['AgeuponOutcome', 'IsDangerous', 'IsPitBull', 'IsWeekend',
             'IsEightAM', 'IsNineAM', 'IsTenAM', 'IsFivePM', 'IsSixPM',
             'IsNamed', 'IsIntact'])

    data = data.loc[:, keep_cols]

    return data


def select_raw_features(data, animal_type):
    """ Selects features present in the raw data."""
    data = extract_features(data, animal_type)

    drop_cols = ['OutcomeSubtype', 'DateTime', 'SexuponOutcome', 'Name']
    data = data.drop(drop_cols, axis=1)
    categorical_columns = ["Breed", 'Month', "Color", 'DayOfWeek', 'Hour']
    for categorical_column in categorical_columns:
        data[categorical_column] = data[categorical_column].astype('category')
    data = pd.get_dummies(data, columns=categorical_columns)

    return data
