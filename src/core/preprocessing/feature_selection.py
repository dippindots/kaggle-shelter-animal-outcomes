'''
Created on Apr 26, 2016

Functions for selecting best features.

@author: Paul Reiners
'''
from core.preprocessing.feature_extraction_scaling \
    import extract_features
import pandas as pd


def select_features(data, animal_type):
    """ Select features that give best score. """
    data = data[data['AnimalType'] == animal_type]
    data = data.drop(['AnimalType'], axis=1)

    data = extract_features(data, animal_type)
    categorical_columns = ["Breed", 'Month', "Color", 'DayOfWeek', 'Hour']
    for categorical_column in categorical_columns:
        data[categorical_column] = data[categorical_column].astype('category')
    data = pd.get_dummies(data, columns=categorical_columns)

    if 'tag' in data.columns:
        keep_cols = ['OutcomeType', 'tag']
    else:
        keep_cols = ['OutcomeType']
    if animal_type == 'Cat':
        keep_cols.extend(
            ['AgeuponOutcome', 'IsDomesticLonghair', 'IsSpring', 'IsJanuary',
             'IsWeekend', 'IsEightAM', 'IsNineAM', 'IsTenAM', 'IsFivePM',
             'IsSixPM', 'IsThreePM', 'IsNamed', 'NamePopularity', 'NameLen',
             'IsIntact', 'Breed_Domestic Longhair', 'Month_1', 'Month_4',
             'Month_5', 'Month_12', 'DayOfWeek_5', 'DayOfWeek_6', 'Hour_8',
             'Hour_9', 'Hour_10', 'Hour_11', 'Hour_14', 'Hour_15', 'Hour_17',
             'Hour_18'])
    else:
        keep_cols.extend(
            ['AgeuponOutcome', 'IsDangerous', 'IsPitBull', 'IsWeekend',
             'IsMidnight', 'IsEightAM', 'IsNineAM', 'IsTenAM', 'IsFivePM',
             'IsSixPM', 'IsNamed', 'NamePopularity', 'NameLen', 'IsIntact',
             'Breed_Border Collie/Akita',
             'Breed_Golden Retriever/Standard Poodle',
             'Breed_Pembroke Welsh Corgi/Brittany', 'Breed_Pit Bull',
             'Breed_Pit Bull/Pit Bull', 'Color_White/Pink', 'DayOfWeek_5',
             'DayOfWeek_6', 'Hour_0', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11',
             'Hour_17', 'Hour_18', 'Hour_19'])

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
