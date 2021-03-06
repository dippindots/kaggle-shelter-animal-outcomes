'''
Created on May 2, 2016

Extract features from raw data and also scale.

@author: Paul Reiners
'''
from numpy import log
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd


def get_data(file_path, tag=None):
    """ Import data from file. """
    dtype = {'Name': str}
    data = pd.read_csv(
        file_path, dtype=dtype, parse_dates=['DateTime'], index_col=0)
    data.Name = data.Name.fillna('')
    if tag:
        data['tag'] = tag

    return data


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
    elif 'Black' in color:
        return 0.5
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
    if breed == 'Pit Bull' or breed == 'American Pit Bull Terrier':
        return 1.0
    elif 'Pit Bull' in breed:
        return 0.5
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


def is_month(actual_month, month_checking_for):
    if actual_month == month_checking_for:
        return 1.0
    else:
        return 0.0


def is_monday(day_of_week):
    if day_of_week == 1:
        return 1.0
    else:
        return 0.0


def is_wednesday(day_of_week):
    if day_of_week == 3:
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


def extract_breed_features(data, animal_type):
    dog_breeds = pd.read_csv('../doc/dog_breeds.csv')
    dog_breeds['American Kennel Club'] = dog_breeds[
        'American Kennel Club'].fillna('')
    data['IsDangerous'] = data['Breed'].apply(is_dangerous)
    data['IsDoodleDog'] = data['Breed'].apply(is_doodle_dog)

    data['IsMix'] = data['Breed'].apply(is_mix)
    data['Breed'] = data['Breed'].apply(remove_mix)

    data['IsPitBull'] = data['Breed'].apply(is_pit_bull)

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

    def is_cat_breed(actual_breed, possible_breed):
        return is_breed(actual_breed, possible_breed, 'Cat')

    def is_dog_breed(actual_breed, possible_breed):
        return is_breed(actual_breed, possible_breed, 'Dog')

    def is_breed(actual_breed, possible_breed, actual_animal_type):
        if animal_type == actual_animal_type:
            if actual_breed == possible_breed:
                return 1.0
            elif possible_breed in actual_breed:
                return 0.5
            else:
                return 0.0
        else:
            return 0.0

    def is_domestic_long_hair(actual_breed):
        return is_cat_breed(actual_breed, 'Domestic Longhair')
    data['IsDomesticLonghair'] = data['Breed'].apply(is_domestic_long_hair)

    def is_domestic_short_hair(actual_breed):
        return is_cat_breed(actual_breed, 'Domestic Shorthair')
    data['IsDomesticShorthair'] = data['Breed'].apply(is_domestic_short_hair)

    def is_siamese(actual_breed):
        return is_cat_breed(actual_breed, 'Siamese')
    data['IsSiamese'] = data['Breed'].apply(is_siamese)

    def is_domestic_short_hair_siamese(actual_breed):
        return is_cat_breed(actual_breed, 'Domestic Shorthair/Siamese')
    data['IsDomesticShorthair_Siamese'] = data[
        'Breed'].apply(is_domestic_short_hair_siamese)

    dog_breeds = ['Border Collie', 'Akita',
                  'Chow Chow', 'Golden Retriever',
                  'Basset Hound', 'Great Pyrenees', 'Chow Chow/Basset Hound']
    for dog_breed in dog_breeds:
        feature_name = "Is" + dog_breed.replace(" ", "").replace("/", "_")
        data[feature_name] = data['Breed'].apply(
            lambda actual_breed: is_dog_breed(actual_breed, dog_breed))

    return data


def get_hour(date_time):
    return date_time.hour


def is_hour(hour, expected_hour):
    if hour == expected_hour:
        return 1.0
    else:
        return 0.0


def extract_date_time_features(data):
    month = data['DateTime'].apply(get_month)
    data['Month'] = month
    data['IsSpring'] = month.apply(is_spring)
    data['DayOfWeek'] = data['DateTime'].apply(get_day_of_week)
    data['IsWeekend'] = data['DayOfWeek'].apply(is_weekend)
    hour = data['DateTime'].apply(get_hour)
    data['Hour'] = hour

    return data


def extract_color_features(data, animal_type):
    data['IsBlack'] = data['Color'].apply(is_black)

    def is_color(actual_color, expected_color, expected_animal_type):
        if animal_type == expected_animal_type:
            if actual_color == expected_color:
                return 1.0
            elif expected_color in actual_color:
                return 0.5
            else:
                return 0.0
        else:
            return 0.0

    def is_cat_color(actual_color, possible_color):
        return is_color(actual_color, possible_color, 'Cat')

    def is_brown_tabby(actual_color):
        return is_cat_color(actual_color, 'Brown Tabby')
    data['IsBrownTabby'] = data['Color'].apply(is_brown_tabby)

    def is_gray(actual_color):
        return is_cat_color(actual_color, 'Gray')
    data['IsGray'] = data['Color'].apply(is_gray)

    def is_brown_tabby_gray(actual_color):
        return is_cat_color(actual_color, 'Brown Tabby/Gray')
    data['IsBrownTabby_Gray'] = data['Color'].apply(is_brown_tabby_gray)

    return data


def get_popularity(name):
    if len(name) == 0:
        return 0.0
    else:
        return 0.0


def extract_name_features(data, animal_type):
    data['IsNamed'] = data['Name'].apply(get_is_named)

    name_counts = data.Name.value_counts(normalize=True)

    def get_popularity(name):
        if len(name) == 0:
            return 0.5
        else:
            return name_counts[name]
    data['NamePopularity'] = data['Name'].apply(get_popularity)

    return data


def extract_age_upon_outcome_features(data):
    data['AgeuponOutcome'] = data['AgeuponOutcome'].apply(preprocess_age)
    data['AgeuponOutcome'] = data['AgeuponOutcome'].fillna(
        data['AgeuponOutcome'].mean())
    data['IsAdult'] = data['AgeuponOutcome'].apply(
        lambda age_in_days: 0 if age_in_days < 365 else 1)

    # Can't take the log of 0.
    data['LogAgeuponOutcome'] = data['AgeuponOutcome'].apply(
        lambda age_in_days: log(age_in_days + 0.01))

    mms = MinMaxScaler()
    data['AgeuponOutcome'] = mms.fit_transform(
        data['AgeuponOutcome'].reshape(-1, 1))
    mms_log = MinMaxScaler()
    data['LogAgeuponOutcome'] = mms_log.fit_transform(
        data['LogAgeuponOutcome'].reshape(-1, 1))

    return data


def extract_features(data, animal_type):
    data = extract_breed_features(data, animal_type)
    data = extract_date_time_features(data, animal_type)
    data = extract_color_features(data, animal_type)
    data = extract_name_features(data, animal_type)
    data = extract_age_upon_outcome_features(data)

    data['IsIntact'] = data['SexuponOutcome'].apply(is_intact)
    data["OutcomeType"] = data["OutcomeType"].astype('category')

    data['IsMale'] = data['SexuponOutcome'].apply(is_male)

    return data


def preprocess_age(age_str):
    days = convert_age_to_days(age_str)

    return days


def convert_age_to_days(age_str):
    if type(age_str) is str:
        parts = age_str.split()
        num = int(parts[0])
        unit = parts[1]
        if 'day' in unit:
            result = num
        elif 'week' in unit:
            result = 7 * num
        elif 'month' in unit:
            result = 30 * num
        elif 'year' in unit:
            result = 365 * num

        return float(result)
    else:
        return np.nan
