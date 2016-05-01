'''
Created on Apr 30, 2016

@author: paulreiners
'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd

from util import get_data, split_data, get_is_named, get_month, is_dog, \
    is_intact, is_male, is_black, is_pit_bull, convert_age_to_days


def preprocess_data(data, animal_type):
    data['IsNamed'] = data['Name'].apply(get_is_named)
    data['Month'] = data['DateTime'].apply(get_month)

    data['IsIntact'] = data['SexuponOutcome'].apply(is_intact)
    data['IsMale'] = data['SexuponOutcome'].apply(is_male)
    if animal_type == "Dog":
        data['IsBlack'] = data['Color'].apply(is_black)
        data['IsPitBull'] = data['Breed'].apply(is_pit_bull)
    data["OutcomeType"] = data["OutcomeType"].astype('category')
    data['AgeuponOutcome'] = data['AgeuponOutcome'].apply(convert_age_to_days)

    drop_cols = ['OutcomeSubtype', 'DateTime', 'SexuponOutcome', 'Name']
    data = data.drop(drop_cols, axis=1)
    categorical_columns = ["Breed", 'Month', "Color"]
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

        k_best = SelectKBest(chi2)
        clf_X_train = k_best.fit_transform(X_train, y_train)

        print "{}".format(X_train.columns[k_best.get_support()])
