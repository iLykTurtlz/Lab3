from abc import ABC, abstractmethod
from DecisionTree import CategoricalDecisionTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


class IncompatibleDimensionsError(Exception):
    def __init__(self, message):
        super().__init__(message)

class NotFittedError(Exception):
    def __init__(self, message):
        super().__init__(message)

class Classifier(ABC):
    """
    This is the abstract base class for classifiers.
    """
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("This method has not been implemented.")
    
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("This method has not been implemented.")


class DecisionTreeClassifier(Classifier):
    """
    This classifier creates a DecisionTree instance when the fit method is called.  
    We may eventually have different types of decision trees.
    If the predict method is called before the fit method, or if asked to fit incompatible data, an error is raised.
    """
    def __init__(self, kind='categorical'):
        super().__init__()
        self.kind = kind
        self.tree = None

    def fit(self, X, y, threshold=0.01, ratio=False):
        if self.kind == 'categorical':
            self.tree = CategoricalDecisionTree()
            self.tree.fit(X, y, threshold, ratio)
        else:
            print("Invalid choice of DecisionTree")

    def predict(self, X):
        return self.tree.predict(X)
    
    def to_json(self, data_file: str, write_file: str = None, return_str: bool = False):
        representation = self.tree.to_json(data_file, write_file, return_str)
        if return_str:
            return representation
        
    def from_json(self, file: str):
        self.tree = CategoricalDecisionTree()
        self.tree.from_json(file)
        
class RandomForestClassifier(Classifier):
    def __init__(self, m: int, k: int, N: int):
        self.m = m
        self.k = k
        self.N = N
    
class Distance:
    def euclidean_dist(x1, x2):
        return np.sqrt(sum((x1 - x2)**2))

    def manhattan_dist(x1, x2):
        return sum(abs(x1-x2))
    
    def cosine_similarity(x1, x2):
        if sum(x1) == 0 or sum(x2) == 0:
            return 0 #not sure what to do here
        return -((x1 @ x2) / ((sum(x1**2))*(sum(x2**2)))) #return the opposite of the cosine similarity to be compatible with sort by increasing value
    


class KNNClassifier(Classifier):
    """
    This classifier stores the data passed in when the fit method is called.
    A variety of distance measures will eventually be possible, but it is Euclidean by default.
    If the predict method is called before the fit method, or if asked to fit incompatible data, an error is raised.
    """
    def __init__(self, k, distance="euclidean"):
        super().__init__()
        self.k = k
        if distance == "euclidean":
            self.dist=Distance.euclidean_dist
        elif distance == "manhattan":
            self.dist=Distance.manhattan_dist
        else:
            raise ValueError
        
    def fit(self, X, y):
        if len(X) != len(y):
            raise Exception("The independent and dependent variable lists must be compatible.")
        self.data = X
        self.labels = y

    def predict(self, X):
        if self.X == None or self.y == None:
            raise NotFittedError(f"{type(self).__name__}.fit(X,y) must be called before this method.")
        if X.shape[1] != self.data.shape[1]:
            raise IncompatibleDimensionsError(f"This {type(self).__name__} can only make predictions on data points of shape {self.data[0].shape}")
        predictions = np.zeros(len(X), dtype=self.labels.dtype)
        for i, x in enumerate(X):
            sorted_idx = np.apply_along_axis(partial(self.dist, x), axis=1, arr=self.data).argsort()
            nearest_neighbors = self.labels[sorted_idx][:self.k]
            values, counts = np.unique(nearest_neighbors, return_counts=True)
            predictions[i] = values[np.argmax(counts)]
        return predictions





