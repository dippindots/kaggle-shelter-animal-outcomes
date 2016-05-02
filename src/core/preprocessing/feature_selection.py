'''
Created on Apr 26, 2016

@author: Paul Reiners
'''
from core.preprocessing.feature_extraction_scaling \
    import extract_features
import pandas as pd


def select_features(data, animal_type):
    extract_features(data, animal_type)

    if 'tag' in data.columns:
        keep_cols = ['OutcomeType', 'tag']
    else:
        keep_cols = ['OutcomeType']
    if animal_type == 'Cat':
        keep_cols.extend(
            ['AgeuponOutcome', 'IsNamed', 'IsIntact', 'IsSpring', 'IsWeekend',
             'IsChristmas'])
    else:
        keep_cols.extend(['AgeuponOutcome', 'IsNamed',
                          'IsIntact', 'IsPitBull', 'IsDangerous', 'IsWeekend',
                          'Toy Group', 'Hound Group'])

    data = data.loc[:, keep_cols]

    return data


def select_raw_features(data, animal_type):
    extract_features(data, animal_type)

    drop_cols = ['OutcomeSubtype', 'DateTime', 'SexuponOutcome', 'Name']
    data = data.drop(drop_cols, axis=1)
    categorical_columns = ["Breed", 'Month', "Color", 'DayOfWeek']
    for categorical_column in categorical_columns:
        data[categorical_column] = data[categorical_column].astype('category')
    data = pd.get_dummies(data, columns=categorical_columns)

    return data