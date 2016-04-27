'''
Created on Apr 26, 2016

@author: paulreiners
'''
import pandas as pd 
import csv

if __name__ == '__main__':
    dtype = {'Name':str}
    test = pd.read_csv('../data/test.csv', dtype=dtype, parse_dates=['DateTime'], index_col=0)
    with open('eggs.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
    for row in test.iterrows():
        pass