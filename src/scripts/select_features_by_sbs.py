'''
Created on Apr 30, 2016

@author: Paul Reiners

Based on code by Sebastian Raschka from his book _Python Machine Learning_.
'''
from sklearn.neighbors import KNeighborsClassifier

from core.preprocessing.feature_extraction_scaling import get_data
from core.preprocessing.feature_selection import select_raw_features
from core.preprocessing.sequential_backward_selection import SBS


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

        knn = KNeighborsClassifier(n_neighbors=2)
        sbs = SBS(knn, k_features=1)
        sbs.fit(X_train.values, y_train.values)
