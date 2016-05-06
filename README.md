# kaggle-shelter-animal-outcomes
Shelter Animal Outcomes Kaggle Competition

## Report
My report is available [here](https://github.com/paul-reiners/kaggle-shelter-animal-outcomes/blob/master/src/report.ipynb).

## Data
The data for this project is available [here](https://www.kaggle.com/c/shelter-animal-outcomes/data).  To run the Python scripts,
you should download these files into a directory named *data* that is a sibling to the 
[*src* directory](https://github.com/paul-reiners/kaggle-shelter-animal-outcomes/tree/master/src).  There
is also another data file that I have created called 
[dog_breeds.csv](https://github.com/paul-reiners/kaggle-shelter-animal-outcomes/blob/master/doc/dog_breeds.csv)
 that needs to be in the [doc directory](https://github.com/paul-reiners/kaggle-shelter-animal-outcomes/tree/master/doc).
 (It will already be there if you have downloaded all the project files from GitHub.)

 ## Software and libraries
 
 * Python 2.7.11 :: Anaconda 4.0.0 (x86_64)
 * scikit-learn 0.17.1 
 * pandas 0.18.0
 * numpy 1.10.4
 * matplotlib 1.5.1
 
 ## Executing scripts
 There are three scripts you might be interested in running:
 * [select_features](./src/scripts/select_features.py)
 * [classifier_comparison](./src/scripts/classifier_comparison.py)
 * [create_submission](./src/scripts/create_submission.py)
 
 Each of these should be run from the [src](./src) directory.  None of them take command-line arguments.
