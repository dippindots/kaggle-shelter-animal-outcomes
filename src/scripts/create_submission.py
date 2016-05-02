'''
Created on Apr 26, 2016

@author: Paul Reiners
'''

from core.evaluation import log_loss
from core.learning.classifiers.random_forest_predictor \
    import RandomForestPredictor
from core.preprocessing.feature_extraction_scaling import get_data
from core.preprocessing.sampling import split_data
from core.util import preprocess_data
import numpy as np


BEST_SCORE = 0.82221


if __name__ == '__main__':
    predictors = {
        'Cat': RandomForestPredictor('Cat'),
        'Dog': RandomForestPredictor('Dog')}
    test_data_sets = {}
    all_predictions_df = None
    all_y_test = None

    # Iterate over AnimalType
    for animal_type in ['Cat', 'Dog']:
        train_data = get_data('../data/train.csv', 'train')
        test_data = get_data('../data/test.csv', 'test')
        all_data = train_data.append(test_data)
        all_data = all_data[all_data['AnimalType'] == animal_type]
        all_data = all_data.drop(['AnimalType'], axis=1)
        all_data = preprocess_data(all_data, animal_type)

        train_data = all_data[all_data['tag'] == 'train']
        train_data = train_data.drop(['tag'], axis=1)
        train_data = train_data.dropna()
        test_data = all_data[all_data['tag'] == 'test']
        test_data = test_data.drop(['OutcomeType', 'tag'], axis=1)
        test_data = test_data.fillna(test_data.mean())
        test_data_sets[animal_type] = test_data

        X_train, y_train, X_test, y_test = split_data(train_data)
        if all_y_test is None:
            all_y_test = y_test.ravel()
        else:
            all_y_test = np.append(all_y_test, y_test.ravel())
        predictor = predictors[animal_type]

        predictor.fit(X_train, y_train)
        predictions_df = predictor.predict(X_test)
        if all_predictions_df is None:
            all_predictions_df = predictions_df
        else:
            all_predictions_df = all_predictions_df.append(predictions_df)

    possible_outcomes = [
        'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    ll = log_loss(
        all_y_test, 'OutcomeType', all_predictions_df, possible_outcomes)

    print "score: %.5f" % ll

    if ll < BEST_SCORE:
        all_test_predictions = None
        # Iterate over AnimalType
        for animal_type in ['Cat', 'Dog']:
            test_data = test_data_sets[animal_type]
            index = test_data.index.values
            predictor = predictors[animal_type]
            test_predictions = predictor.predict(test_data)
            test_predictions['ID'] = index
            test_predictions = test_predictions.set_index('ID')
            if all_test_predictions is None:
                all_test_predictions = test_predictions
            else:
                all_test_predictions = all_test_predictions.append(
                    test_predictions)
        all_test_predictions = all_test_predictions.sort_index()
        columns = [
            'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
        all_test_predictions.to_csv('../submissions/my_submission.csv',
                                    index=True, columns=columns)
