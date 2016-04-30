'''
Created on Apr 26, 2016

@author: Paul Reiners
'''

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from classifiers.random_forest_predictor import RandomForestPredictor
from util import measure_log_loss_of_predictor, get_data, split_data, \
    preprocess_data


BEST_SCORE = 0.82221


if __name__ == '__main__':
    train_data = get_data('../data/train.csv', 'train')
    test_data = get_data('../data/test.csv', 'test')
    all_data = train_data.append(test_data)
    all_data = preprocess_data(all_data)

    train_data = all_data[all_data['tag'] == 'train']
    train_data = train_data.drop(['tag'], axis=1)
    train_data = train_data.dropna()
    test_data = all_data[all_data['tag'] == 'test']
    test_data = test_data.drop(['OutcomeType', 'tag'], axis=1)
    test_data = test_data.fillna(test_data.mean())

    X_train, y_train, X_test, y_test = split_data(train_data)
    k_best = SelectKBest(chi2, k=10)
    X_train = k_best.fit_transform(X_train, y_train)
    X_test = k_best.transform(X_test)

    predictor = RandomForestPredictor()
    ll = measure_log_loss_of_predictor(
        X_train, y_train, X_test, y_test, predictor)
    print "score: %.5f" % ll

    if ll < BEST_SCORE:
        test_data = k_best.transform(test_data)
        test_predictions = predictor.predict(test_data)
        columns = ['ID', 'Adoption', 'Died',
                   'Euthanasia', 'Return_to_owner', 'Transfer']
        test_predictions.to_csv('../submissions/my_submission.csv',
                                index=False, columns=columns)
