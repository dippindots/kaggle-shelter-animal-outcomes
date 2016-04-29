'''
Created on Apr 26, 2016

@author: Paul Reiners
'''

from classifiers.decision_tree_predictor import DecisionTreePredictor
import numpy as np
import pandas as pd
from util import measure_log_loss_of_predictor, get_data, split_data


BEST_SCORE = 0.94061


def clean_data(data, is_test=False):
    drop_cols = ['OutcomeSubtype', 'Name', 'DateTime']
    data = data.drop(drop_cols, axis=1)
    categorical_columns = [
        "OutcomeType", "AnimalType", "SexuponOutcome", "Breed", "Color"]
    for categorical_column in categorical_columns:
        data[categorical_column] = data[categorical_column].astype('category')
    data['AgeuponOutcome'] = data['AgeuponOutcome'].apply(convert_age_to_days)

    data = pd.get_dummies(
        data, columns=["AnimalType", "SexuponOutcome", "Breed", "Color"])

    return data


def convert_age_to_days(age_str):
    if type(age_str) is str:
        parts = age_str.split()
        num = int(parts[0])
        unit = parts[1]
        if 'day' in unit:
            return num
        elif 'week' in unit:
            return 7 * num
        elif 'month' in unit:
            return 30 * num
        elif 'year' in unit:
            return 365 * num
    else:
        return np.nan


if __name__ == '__main__':
    train_data = get_data('../data/train.csv', 'train')
    test_data = get_data('../data/test.csv', 'test')
    all_data = train_data.append(test_data)
    all_data = clean_data(all_data)

    train_data = all_data[all_data['tag'] == 'train']
    train_data = train_data.drop(['tag'], axis=1)
    train_data = train_data.dropna()
    test_data = all_data[all_data['tag'] == 'test']
    test_data = test_data.drop(['OutcomeType', 'tag'], axis=1)
    test_data = test_data.fillna(test_data.mean())

    X_train, y_train, X_test, y_test = split_data(train_data)

    predictor = DecisionTreePredictor()
    ll = measure_log_loss_of_predictor(
        X_train, y_train, X_test, y_test, predictor)
    print "score: %.5f" % ll

    if ll < BEST_SCORE:
        test_predictions = predictor.predict(test_data)
        columns = ['ID', 'Adoption', 'Died',
                   'Euthanasia', 'Return_to_owner', 'Transfer']
        test_predictions.to_csv('../submissions/my_submission.csv',
                                index=False, columns=columns)
