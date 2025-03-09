############################## Importing dependencies ##########################

## The cwd path should be the path to the folder in which the .py file containing code is present
## All other supporting functions are also present in the same directory


from LogisticRegression import logistic_regression
from ClassificationMetrics import classification_metrics
from CustomCountVectorizer import CustomCountVectorizer
import os
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix  ## for converting dense matrix to sparse matrix

#**********************************************************************************


############################ Defining helper functions ##############################

## Function to split training and test datasets
def train_test_splitter(n, test_size = 0.3):
    """
    Function to randomly split dataset into train and test datasets. Return indices for splitting.
    n: no. of samples
    test_size : fraction of samples to be put in the test set
    """

    np.random.seed(23)  ## setting random seed for re-producability
    test_n = int(test_size * n)
    rand_ind = np.random.permutation(n)
    test_ind = rand_ind[:test_n]
    train_ind = rand_ind[test_n:]
    return train_ind, test_ind

## Creating function to convert y array to sparse
def y_to_csr(y):
    """
    Function to convert y array to csr format
    """
    row = []
    col = [0]*y.shape[0]
    data = []
    for ind, val in enumerate(y):
        row.append(ind)
        data.append(val)
    y_csr = csr_matrix((data, (row, col)), shape = (y.shape[0], 1))
    return y_csr

#**********************************************************************************

############################ Defining main training function ##############################

def training_logreg():
    """
    Main function for training logistic regression classifier. Returns fitted model and count-vectorizer objects.
    Takes pre-processed emails from .csv files saved as "processed_emails.csv"
    """
    
    ## Deleting preivously created logreg model pickle file
    if os.path.exists(os.path.join(os.getcwd(), "logreg_c.pkl")):
        os.remove(os.path.join(os.getcwd(), "logreg_c.pkl"))
    
    ## Deleting preivously created model count vectorizer file
    if os.path.exists(os.path.join(os.getcwd(), "cv.pkl")):
        os.remove(os.path.join(os.getcwd(), "cv.pkl"))

    df = pd.read_csv(os.path.join(os.getcwd(), "processed_emails.csv"))  ## loading pre-processed emails from csv file
    df = df.dropna()

    X = np.array(df["content"])   ## Feature matrix
    y = np.array(list(df["label"]))  ## labels

    ## checking if all the entries in the training set are of string type (and not nan)
    for i, text in enumerate(X):
        if not isinstance(text, str):
            print(f"The following index has non-string entry: {i}")

    ## splitting the dataset into train and test datasets randomly
    ## test dataset has 30% of the total emails
    train_ind, test_ind = train_test_splitter(len(df), 0.3)
    X_train, y_train = X[train_ind], y[train_ind]
    X_test, y_test = X[test_ind], y[test_ind]

    cv = CustomCountVectorizer(min_df = 2)  ## fitting custom count vectorizer on training data
    cv.fit(X_train)

    with open(os.path.join(os.getcwd(), "cv.pkl"), "wb") as file: ## saving the fitted cv for testing using pickle
        pickle.dump(cv, file = file)

    print(f"Words in vocabulary: {len(cv.vocabulary_)}")

    X_train_cv = cv.transform(X_train)  ## Fitting CountVectorizer on the training dataset
    X_test_cv = cv.transform(X_test)  ## Transforming test dataset

    y_train_csr = y_to_csr(y_train)   ## converting y_train to a sparse array

    ## Fitting  logistic regression
    logreg_c = logistic_regression(threshold=0.5, step_size_multiplier=80, max_iter=5*10**3)  ## fitting the logistic regression model
    grads = logreg_c.fit(X_train_cv, y_train_csr)

    with open(os.path.join(os.getcwd(), "logreg_c.pkl"), "wb") as file: ## saving the fitted cv for testing using pickle
        pickle.dump(logreg_c, file = file)


    ## calculating the metrics on the training dataset
    print("Logistic Regression performance on training dataset:")
    classification_metrics(y_train, logreg_c.predict(X_train_cv))

    ## calculating the metrics on the test dataset
    print("Logistic Regression performance on test dataset:")
    classification_metrics(y_test, logreg_c.predict(X_test_cv))

    return logreg_c, cv

print("Training.py file execution started")
training_logreg()