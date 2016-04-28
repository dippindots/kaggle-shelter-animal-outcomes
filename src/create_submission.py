'''
Created on Apr 26, 2016

See [Classifier comparison]
(http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
for possible classifiers.

Best predictors:
    Date        Type                MyLLScore  KaggleLLScore  GithubTag
    ======================================================================
    04/27/2016  BaseLinePredictor    20.61577       20.25113  Submission00
    04/27/2016  KNeighborsPredictor  14.37823       13.94696  Submission01
    04/27/2016  KNeighborsPredictor   5.07153        5.20698  Submission02

@author: Paul Reiners
'''
from sklearn.cross_validation import train_test_split

from k_neighbors_predictor import KNeighborsPredictor
import numpy as np
import pandas as pd
from util import log_loss


BEST_SCORE = 5.07153


def get_data(file_path, tag):
    dtype = {'Name': str}
    data = pd.read_csv(
        file_path, dtype=dtype, parse_dates=['DateTime'], index_col=0)
    data['tag'] = tag

    return data


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

    X = train_data.drop(['OutcomeType'], axis=1)
    y = train_data['OutcomeType']
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    predictor = KNeighborsPredictor()
    predictor.fit(X_train, y_train)
    predictions_df = predictor.predict(X_test)

    possible_outcomes = [
        'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    ll = log_loss(y_test, 'OutcomeType', predictions_df, possible_outcomes)
    print "score: %.5f" % ll

    if ll < BEST_SCORE:
        test_predictions = predictor.predict(test_data)
        columns = ['ID', 'Adoption', 'Died',
                   'Euthanasia', 'Return_to_owner', 'Transfer']
        test_predictions.to_csv('../submissions/my_submission.csv',
                                index=False, columns=columns)
