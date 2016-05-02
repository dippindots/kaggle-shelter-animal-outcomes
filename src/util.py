'''
Created on Apr 26, 2016

@author: Paul Reiners
'''
from math import log

from sklearn.cross_validation import train_test_split

import numpy as np
import pandas as pd


def log_loss(truths, label_col_name, predictions_df, possible_labels):
    ''' Function to measure log loss of a prediction.

    Parameters
    ==========
    truths          : numpy.ndarray
                      the ground truth
    label_col_name  : str
                      the name of the column you're trying to predict
    predictions_df  : pandas.core.frame.DataFrame
                      your predictions
    possible_labels : list
                      possible labels'''
    n = len(truths)
    total = 0.0
    for i in range(n):
        truth = truths[i]
        for possible_label in possible_labels:
            if truth == possible_label:
                y = 1
            else:
                y = 0
            prediction = predictions_df.iloc[i]
            p = prediction[truth]
            p = max(min(p, 1 - 1e-15), 1e-15)
            total += y * log(p)

    return -1.0 / n * total


def get_data(file_path, tag=None):
    dtype = {'Name': str}
    data = pd.read_csv(
        file_path, dtype=dtype, parse_dates=['DateTime'], index_col=0)
    data.Name = data.Name.fillna('')
    data.SexuponOutcome = data.SexuponOutcome.fillna('')
    if tag:
        data['tag'] = tag

    return data


def split_data(train_data):
    X = train_data.drop(['OutcomeType'], axis=1)
    y = train_data['OutcomeType']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, y_train, X_test, y_test


def get_is_named(name):
    if len(name) > 0:
        return 1.0
    else:
        return 0.0


def get_month(date_time):
    return date_time.month


def get_day_of_week(date_time):
    return date_time.dayofweek


def is_dog(animal_type):
    if animal_type == 'Dog':
        return 1.0
    else:
        return 0.0


def is_intact(sex_upon_outcome):
    if 'Intact' in sex_upon_outcome:
        return 1.0
    elif 'Neutered' in sex_upon_outcome or 'Spayed' in sex_upon_outcome:
        return 0.0
    else:
        return 0.5


def is_black(color):
    if color == 'Black':
        return 1.0
    else:
        return 0.0


def is_male(sex_upon_outcome):
    if 'Male' in sex_upon_outcome:
        return 1.0
    elif 'Female' in sex_upon_outcome:
        return 0.0
    else:
        return 0.5


def is_pit_bull(breed):
    if 'Pit Bull' in breed:
        return 1.0
    else:
        return 0.0


def is_golden_retriever(breed):
    if 'Golden Retriever' in breed:
        return 1.0
    else:
        return 0.0


def is_doodle_dog(breed):
    if 'Poodle' in breed and ('Labrador' in breed or 'Retriever' in breed):
        return 1.0
    else:
        return 0.0


def is_spring(month):
    if month == 4 or month == 5:
        return 1.0
    elif month == 3 or month == 6:
        return 0.5
    else:
        return 0.0


def is_christmas(month):
    if month == 12:
        return 1.0
    else:
        return 0.0


def is_weekend(day_of_week):
    if day_of_week == 5 or day_of_week == 6:
        return 1.0
    else:
        return 0.0


def is_dangerous(breed):
    dangerous_breeds = [
        'Great Dane', 'Boxer', 'Wolf Hybrid', 'Malamute', 'Husky', 'Mastiff',
        'Doberman Pinscher', 'Doberman Pinsch', 'German Shepherd',
        'Rottweiler', 'Pit Bull']
    for dangerous_breed in dangerous_breeds:
        if dangerous_breed in breed:
            return 1.0
    return 0.0


def is_mix(breed):
    if "Mix" in breed:
        return 1.0
    else:
        return 0.0


def remove_mix(breed):
    if breed.endswith('Mix'):
        return breed[:-len('Mix')].strip()
    else:
        return breed


def commmon_preprocess_data(data, animal_type):
    dog_breeds = pd.read_csv('../doc/dog_breeds.csv')
    dog_breeds['American Kennel Club'] = dog_breeds[
        'American Kennel Club'].fillna('')

    data['AgeuponOutcome'] = data['AgeuponOutcome'].apply(preprocess_age)

    data['IsNamed'] = data['Name'].apply(get_is_named)
    data['IsIntact'] = data['SexuponOutcome'].apply(is_intact)
    data["OutcomeType"] = data["OutcomeType"].astype('category')
    month = data['DateTime'].apply(get_month)
    data['IsPitBull'] = data['Breed'].apply(is_pit_bull)
    data['IsDangerous'] = data['Breed'].apply(is_dangerous)
    data['IsBlack'] = data['Color'].apply(is_black)
    data['IsGoldenRetriever'] = data[
        'Breed'].apply(is_golden_retriever)
    data['IsDoodleDog'] = data['Breed'].apply(is_doodle_dog)
    data['IsSpring'] = month.apply(is_spring)
    data['IsChristmas'] = month.apply(is_christmas)

    data['IsMale'] = data['SexuponOutcome'].apply(is_male)

    data['IsMix'] = data['Breed'].apply(is_mix)
    data['Breed'] = data['Breed'].apply(remove_mix)

    data['Month'] = month
    data['DayOfWeek'] = data['DateTime'].apply(get_day_of_week)
    data['IsWeekend'] = data['DayOfWeek'].apply(is_weekend)

    def is_dog_type(breed, dog_type):
        if animal_type == 'Cat':
            return 0.0
        else:
            parts = breed.split('/')
            for part in parts:
                dog_breed = dog_breeds.loc[dog_breeds['Breed'] == part]
                if len(dog_breed) < 1:
                    return 0.0
                group = dog_breed.iloc[0]['American Kennel Club']
                group_parts = group.split(',')
                for group_part in group_parts:
                    if group_part.strip() == dog_type:
                        return 1.0
            return 0.0

    groups = ['Sporting Group', 'Toy Group', 'Herding Group', 'Hound Group',
              'Non-Sporting Group', 'Terrier Group', 'Working Group']
    for group in groups:
        data[group] = data['Breed'].apply(
            lambda breed: is_dog_type(breed, group))

    def is_popular_dog_breed(breed):
        # http://www.akc.org/news/the-most-popular-dog-breeds-in-america/
        popular_breeds = [
            'Labrador Retriever', 'German Shepherd', 'Golden Retriever',
            'American Bulldog', 'Beagle', 'French Bulldog',
            'Yorkshire Terrier', 'Poodle']
        if animal_type == 'Cat':
            return 0.0
        else:
            for popular_breed in popular_breeds:
                if popular_breed in breed:
                    return 1.0
            return 0.0
    data['IsPopularDog'] = data['Breed'].apply(is_popular_dog_breed)


def preprocess_data(data, animal_type):
    commmon_preprocess_data(data, animal_type)

    if 'tag' in data.columns:
        keep_cols = ['OutcomeType', 'tag']
    else:
        keep_cols = ['OutcomeType']
    if animal_type == 'Cat':
        keep_cols.extend(
            ['AgeuponOutcome', 'IsNamed', 'IsIntact', 'IsSpring', 'IsWeekend',
             'IsChristmas'])
    else:
        keep_cols.extend(['AgeuponOutcome', 'IsNamed',
                          'IsIntact', 'IsPitBull', 'IsDangerous', 'IsWeekend',
                          'Toy Group', 'Hound Group'])

    data = data.loc[:, keep_cols]

    return data


def preprocess_age(age_str):
    days = convert_age_to_days(age_str)
    if np.isnan(days):
        return days
    else:
        return log(1 + days)


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
