'''
Created on May 2, 2016

@author: Paul Reiners
'''
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd


def get_data(file_path, tag=None):
    dtype = {'Name': str}
    data = pd.read_csv(
        file_path, dtype=dtype, parse_dates=['DateTime'], index_col=0)
    data.Name = data.Name.fillna('')
    data.SexuponOutcome = data.SexuponOutcome.fillna('')
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


def is_golden_retriever(breed):
    if breed == 'Golden Retriever':
        return 1.0
    elif 'Golden Retriever' in breed:
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


def is_christmas(month):
    if month == 12:
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
    data['IsGoldenRetriever'] = data[
        'Breed'].apply(is_golden_retriever)

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

    def is_border_collie(actual_breed):
        return is_dog_breed(actual_breed, 'Border Collie')
    data['IsBorderCollie'] = data['Breed'].apply(is_border_collie)

    def is_akita(actual_breed):
        return is_dog_breed(actual_breed, 'Akita')
    data['IsAkita'] = data['Breed'].apply(is_akita)

    return data


def get_hour(date_time):
    return date_time.hour


def extract_date_time_features(data, animal_type):
    month = data['DateTime'].apply(get_month)
    data['IsSpring'] = month.apply(is_spring)
    data['IsChristmas'] = month.apply(is_christmas)
    data['Month'] = month
    data['DayOfWeek'] = data['DateTime'].apply(get_day_of_week)
    data['IsWeekend'] = data['DayOfWeek'].apply(is_weekend)
    data['IsMonday'] = data['DayOfWeek'].apply(is_monday)
    data['IsWednesday'] = data['DayOfWeek'].apply(is_wednesday)
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

    return data


def extract_features(data, animal_type):
    data = extract_breed_features(data, animal_type)
    data = extract_date_time_features(data, animal_type)
    data = extract_color_features(data, animal_type)

    data['AgeuponOutcome'] = data['AgeuponOutcome'].apply(preprocess_age)

    data['IsNamed'] = data['Name'].apply(get_is_named)
    data['IsIntact'] = data['SexuponOutcome'].apply(is_intact)
    data["OutcomeType"] = data["OutcomeType"].astype('category')

    data['IsMale'] = data['SexuponOutcome'].apply(is_male)

    data = data.fillna(data.mean())

    mms = MinMaxScaler()
    data['AgeuponOutcome'] = mms.fit_transform(
        data['AgeuponOutcome'].reshape(-1, 1))

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
            return num
        elif 'week' in unit:
            return 7 * num
        elif 'month' in unit:
            return 30 * num
        elif 'year' in unit:
            return 365 * num
    else:
        return np.nan
