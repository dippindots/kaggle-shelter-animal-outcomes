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
