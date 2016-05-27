'''
Created on May 26, 2016

@author: Paul Reiners
'''
import sklearn.preprocessing

import numpy as np
import pandas as pd


def get_features_and_labels():
    features_df = pd.DataFrame.from_csv(
        "../data/train.csv", parse_dates=['DateTime'])
    labels_df = pd.DataFrame(features_df['OutcomeType'])
    labels_df['OutcomeType'] = labels_df['OutcomeType'].astype('category')
    features_df = features_df.drop(['OutcomeType'], axis=1)

    return (features_df, labels_df)


def get_names_of_columns_to_transform():
    return ['AnimalType', 'SexuponOutcome', 'Breed', 'Color', 'Month',
            'DayOfWeek', 'Hour']


def hot_encoder(df, column_name):
    column = df[column_name].tolist()
    # needs to be an N x 1 numpy array
    column = np.reshape(column, (len(column), 1))
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(column)
    new_column = enc.transform(column).toarray()
    # making titles for the new columns, and appending them to dataframe
    for ii in range(len(new_column[0])):
        this_column_name = column_name + "_" + str(ii)
        df[this_column_name] = new_column[:, ii]
    return df
