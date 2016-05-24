'''
Created on Apr 26, 2016

Functions for selecting best features.

@author: Paul Reiners
'''
from core.preprocessing.feature_extraction_scaling \
    import extract_features
import pandas as pd


def select_features(data, animal_type, is_adult):
    """ Select features that give best score. """
    data = data[data['AnimalType'] == animal_type]
    data = data.drop(['AnimalType'], axis=1)

    data = extract_features(data, animal_type)

    data = data[data['IsAdult'] == is_adult]
    data = data.drop(['IsAdult'], axis=1)

    categorical_columns = ["Breed", 'Month', "Color", 'DayOfWeek', 'Hour']
    for categorical_column in categorical_columns:
        data[categorical_column] = data[categorical_column].astype('category')
    data = pd.get_dummies(data, columns=categorical_columns)

    if 'tag' in data.columns:
        keep_cols = ['OutcomeType', 'tag']
    else:
        keep_cols = ['OutcomeType']
    if animal_type == 'Cat':
        if not is_adult:
            keep_cols.extend(
                ['IsSpring', 'IsWeekend', 'IsNamed', 'NamePopularity',
                 'IsIntact',
                 'Month_4', 'Color_Agouti', 'Hour_8', 'Hour_9', 'Hour_17'])
        else:
            keep_cols.extend(
                ['IsWeekend', 'IsNamed', 'NamePopularity', 'IsIntact',
                 'DayOfWeek_5', 'DayOfWeek_6', 'Hour_9', 'Hour_10', 'Hour_17',
                 'Hour_18'])
    else:
        if not is_adult:
            keep_cols.extend(
                ['IsPitBull', 'IsNamed', 'NamePopularity', 'IsIntact',
                 'Breed_Pit Bull', 'Breed_Pit Bull/Pit Bull', 'Hour_11',
                 'Hour_14',
                 'Hour_17', 'Hour_18'])
        else:
            keep_cols.extend(
                ['IsPitBull', 'IsWeekend', 'NamePopularity', 'IsIntact',
                 'Breed_Golden Retriever/Standard Poodle',
                 'Breed_Pembroke Welsh Corgi/Brittany', 'Breed_Pit Bull',
                 'Color_White/Pink', 'Hour_0', 'Hour_11'])

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
