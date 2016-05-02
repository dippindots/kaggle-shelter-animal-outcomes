'''
Created on Apr 30, 2016

@author: Paul Reiners
'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from core.preprocessing.feature_extraction_scaling import get_data
from core.util import split_data, commmon_preprocess_data
import pandas as pd


def preprocess_data(data, animal_type):
    commmon_preprocess_data(data, animal_type)

    drop_cols = ['OutcomeSubtype', 'DateTime', 'SexuponOutcome', 'Name']
    data = data.drop(drop_cols, axis=1)
    categorical_columns = ["Breed", 'Month', "Color", 'DayOfWeek']
    for categorical_column in categorical_columns:
        data[categorical_column] = data[categorical_column].astype('category')
    data = pd.get_dummies(data, columns=categorical_columns)

    return data


if __name__ == '__main__':
    possible_outcomes = [
        'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    # Iterate over AnimalType
    for animal_type in ['Cat', 'Dog']:
        train_data = get_data('../data/train.csv')
        print animal_type
        train_data = train_data[train_data['AnimalType'] == animal_type]
        train_data = train_data.drop(['AnimalType'], axis=1)
        train_data = preprocess_data(train_data, animal_type)
        train_data = train_data.dropna()
        X_train, y_train, X_test, y_test = split_data(train_data)

        k_best = SelectKBest(chi2, k=10)
        clf_X_train = k_best.fit_transform(X_train, y_train)

        print "{}".format(X_train.columns[k_best.get_support()])
