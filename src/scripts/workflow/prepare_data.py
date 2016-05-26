'''
Created on May 25, 2016

@author: Paul Reiners

Based on
["Workflows in Python: Getting data ready to build models"]
(https://civisanalytics.com/blog/data-science/2015/12/17/workflows-in-python-getting-data-ready-to-build-models/)
by Katie Malone
'''
import numpy as np
import pandas as pd


if __name__ == '__main__':
    features_df = pd.DataFrame.from_csv("../data/train.csv")
    labels_df = pd.DataFrame(features_df['OutcomeType'])
    features_df = features_df.drop(['OutcomeType'], axis=1)
    print(labels_df.head(20))

    def label_map(y):
        if y == "Adoption":
            return 4
        elif y == "Died":
            return 3
        elif y == "Euthanasia":
            return 2
        elif y == "Return_to_owner":
            return 1
        else:
            # Transfer
            return 0
    labels_df = labels_df.applymap(label_map)
    print(labels_df.head())

    def transform_feature(df, column_name):
        unique_values = set(df[column_name].tolist())
        transformer_dict = {}
        for ii, value in enumerate(unique_values):
            transformer_dict[value] = ii

        def label_map(y):
            return transformer_dict[y]
        df[column_name] = df[column_name].apply(label_map)
        return df

    # list of column names indicating which columns to transform;
    # this is just a start!  Use some of the print( labels_df.head() )
    # output upstream to help you decide which columns get the
    # transformation
    names_of_columns_to_transform = [
        'AnimalType', 'SexuponOutcome', 'Breed', 'Color']
    for column in names_of_columns_to_transform:
        features_df = transform_feature(features_df, column)

    print(features_df.head())

    # remove the "date_recorded" column--we're not going to make use
    # of time-series data today
    features_df.drop("DateTime", axis=1, inplace=True)

    print(features_df.columns.values)

    X = features_df.as_matrix()
    y = labels_df["OutcomeType"].tolist()
