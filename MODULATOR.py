import numpy as np
import pandas as pd
import warnings
from enum import Enum

class Distances(Enum):
    cosine_similarity = "cosine similarity"
    cosine = "cosine"
    euclidean = "euclidean"
    minkowski = "minkowski"
    correlation = "correlation"

def _validate_data(X,y)->None:
    """
    Checks data input to be correct.
    
    Parameters
    -------------------
    X: Any
    y: Any

    Returns
    -------------------
    None

    Raises
    -------------------
    Exception: List, pandas data frame or numpy array expected
    Exception: 2D array like on X expected
    Exception: Unidimensional array on y expected
    Exception: Data conversion warning, y shape expected (n_samples,) but vector column received
    Exception: At least 2 classes needed
    Exception: X's n_sample != y'n n_sample

    """
    #Check the data to be list, pandas data frame or numpy array
    try:
         X_c = np.array(X)
         y_c = np.array(y)
    except:
         raise Exception("List, pandas data frame or numpy array expected")

    #Checks data shape be correct
    if len(X_c.shape) != 2:
        raise Exception("2D array like on X expected")
    
    if len(y_c.shape) > 2:
        raise Exception("Unidimensional array on y expected")
    
    if len(y_c.shape) != 1 and (y_c.shape[0] != 1 and y_c.shape[1] != 1):
        raise Exception("Unidimensional array on y expected")
    
    if len(y_c.shape) != 1:
        warnings.warn("Data conversion warning, y shape expected (n_samples,) but vector column received")
        flag = True
    #Checks number of classes(minimum 2)
    if len(np.unique(y_c)) < 2:
        raise Exception("At least 2 classes needed")
    
    #Checks X _samples same y n_samples
    y_c = np.ravel(y_c)
    if X_c.shape[0] != y_c.shape[0]:
        raise Exception("X's n_sample != y'n n_sample")

class ModulatorFM():
    """
    Modulator class, this class modulates in FM the entries depending in their classes.

    Parameters
    ----------------
    k: float
        Deviation constant.
    """
    def __init__(self,k:float,constant_vector=None,distance:[] = "minkowski") -> None:
        self.k_ = k

    def _split_data(self,X,y)->dict:
        """
        Splits X on differents arrays grouped by the class.

        Parameters
        ---------------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Retuns
        --------------
        dict:
            Dictionary with matrix group by classes.
        """
        X_c = np.array(X)
        y_c = np.ravel(y)
        classes = np.unique(y)
        splited_data = dict()
        for i in range(X.shape[0]):
            try:
                splited_data[y_c[i]] = np.vstack((splited_data[y_c[i]],X_c[i,:]))
            except:
                splited_data[y_c[i]] = X_c[i,:]
        return splited_data,classes

    def _calculate_distance(self,X):
        pass
    
    def _modulate_vectors(self,X):
        pass

    def fit(self,X,y):
        pass

    def fit_transform(self,X,y):
        pass

    def transform(self,X):
        pass

print("minkowski" in Distances)