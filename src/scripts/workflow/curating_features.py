'''
Created on May 25, 2016

@author: Paul Reiners

Based on
["Workflows in Python: Curating Features and Thinking Scientifically about
  Algorithms"]
(https://civisanalytics.com/blog/data-science/2015/12/23/workflows-in-python-curating-features-and-thinking-scientifically-about-algorithms/)
by Katie Malone
'''
import sklearn.preprocessing
import numpy as np


if __name__ == '__main__':
    def hot_encoder(df, column_name):
        column = df[column_name].tolist()
        # needs to be an N x 1 numpy array
        column = np.reshape(column, (len(column), 1))
        enc = sklearn.preprocessing.OneHotEncoder()
        enc.fit(column)
        new_column = enc.transform(column).toarray()
        column_titles = []
        # making titles for the new columns, and appending them to dataframe
        for ii in range(len(new_column[0])):
            this_column_name = column_name + "_" + str(ii)
            df[this_column_name] = new_column[:, ii]
        return df
