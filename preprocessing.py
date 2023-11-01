import pandas as pd
import numpy as np
from pandas.api.types import is_any_real_numeric_dtype


def get_metadata(D: pd.core.frame.DataFrame)->tuple[str, np.ndarray]:
    """Returns the class name and the vector of domain size for each column except the class column"""
    class_name = D.iloc[1,0]
    domain_size_vector = D.iloc[0,:]
    if domain_size_vector.hasnans: #this is just for the mushroom dataset, which leaves out the domain size of the class column!
        domain_size_vector = domain_size_vector.dropna()
    else:
        without_class = D.loc[:,D.columns != class_name]
        domain_size_vector = without_class.iloc[0,:]
    if len(domain_size_vector) != D.shape[1] - 1:
        raise pd.errors.ParserError("Metadata unreadable")
    return class_name, np.array(domain_size_vector, dtype=int)

def get_data(D: pd.core.frame.DataFrame)->tuple[pd.core.frame.DataFrame, pd.core.series.Series]:
    """Returns a dataset X and a list of class labels y"""
    class_name, domain_size_vector = get_metadata(D)
    without_metadata = D.drop(index=[0,1])
    X = without_metadata.loc[:,without_metadata.columns != class_name]
    y = without_metadata[class_name].astype("str")
    #assert(len(X.columns) == len(domain_size_vector))
    for col_name, domain_size in zip(X.columns, domain_size_vector):
        if domain_size == 0:
            X.loc[:,col_name] = pd.to_numeric(X[col_name], errors='coerce')
        else:
            X.loc[:,col_name] = X[col_name].astype("str")
    X = X.replace('?', np.nan) #other values to be interpreted as null?
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y

def normalize_column(X):
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    return np.nan_to_num((X - mins) / (maxs - mins))

def normalize(D: pd.core.frame.DataFrame)->pd.core.frame.DataFrame:
    for col in D.columns:
        if is_any_real_numeric_dtype(D[col]):
            D[col] = normalize_column(D[col])
    return D
        


