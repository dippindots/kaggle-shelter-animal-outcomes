'''
Created on Apr 30, 2016

@author: Paul Reiners

Based on code by Sebastian Raschka from his book _Python Machine Learning_.
'''
from sklearn.ensemble import RandomForestClassifier

from core.preprocessing.feature_extraction_scaling import get_data
from core.preprocessing.feature_selection import select_raw_features
import numpy as np


if __name__ == '__main__':
    possible_outcomes = [
        'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
    # Iterate over AnimalType
    for animal_type in ['Cat', 'Dog']:
        train_data = get_data('../data/train.csv')
        print animal_type
        train_data = train_data[train_data['AnimalType'] == animal_type]
        train_data = train_data.drop(['AnimalType'], axis=1)
        train_data = select_raw_features(train_data, animal_type)
        train_data = train_data.dropna()
        train_data['OutcomeType'] = train_data['OutcomeType'].astype(str)
        X_train = train_data.drop(['OutcomeType'], axis=1)
        y_train = train_data['OutcomeType']

        feat_labels = train_data.columns[1:]
        forest = RandomForestClassifier(n_estimators=10000,
                                        random_state=0,
                                        n_jobs=-1)
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(X_train.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30,
                                    feat_labels[indices[f]],
                                    importances[indices[f]]))
