'''
Created on Apr 26, 2016

@author: Paul Reiners
'''
from core.preprocessing.feature_extraction_scaling \
    import extract_features


def preprocess_data(data, animal_type):
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
