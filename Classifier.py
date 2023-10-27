from abc import ABC, abstractmethod
from DecisionTree import CategoricalDecisionTree, CompleteDecisionTree
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
    def __init__(self, kind='complete', threshold=0.01, ratio=False):
        super().__init__()
        self.kind = kind
        self.tree = None
        self.threshold=threshold
        self.ratio = ratio

    def fit(self, X, y):
        if self.kind == 'categorical':
            self.tree = CategoricalDecisionTree()
            self.tree.fit(X, y, self.threshold, self.ratio)
        elif self.kind == 'complete':
            self.tree = CompleteDecisionTree()
            self.tree.fit(X, y, self.threshold, self.ratio)
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
    def __init__(self, num_attributes: int, num_data_points: int, num_trees: int, threshold=0.01, ratio=False):
        self.num_attributes = num_attributes
        self.num_data_points = num_data_points
        self.num_trees = num_trees
        self.forest = None
        self.threshold = threshold
        self.ratio = ratio

    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0])
        # X = X.reindex(drop=True) #for safety.  They must have the SAME index
        # y = y.reindex(drop=True)

        self.forest = []
        for i in range(self.num_trees):
            cols = np.random.choice(X.columns, self.num_attributes, replace=False)
            X_subset = X[cols]
            X_sample = X_subset.sample(n=self.num_data_points)
            #y_sample = y.iloc[X_sample.index]
            y_sample = y.drop(index=y.index.difference(X_sample.index))
            tree = CompleteDecisionTree()
            tree.fit(X_sample, y_sample, self.threshold, self.ratio) #modify threshold and ratio?
            self.forest.append(tree)

    def predict(self, X):
        if self.forest is None:
            return NotFittedError(f"{type(self).__name__}.fit(X,y) must be called before this method.")
        prediction_list = []
        #votes = dict()
        for tree in self.forest:
            predictions = tree.predict(X)
            prediction_list.append(predictions)
        result = []
        for i in range(len(X)):
            votes = np.array([pred[i] for pred in prediction_list])
            values, counts = np.unique(votes, return_counts=True)
            result.append(values[np.argmax[counts]])
        return result


        

        # for x in X:
        #     for tree in self.forest:
        #         vote = tree.predict(x)
        #         votes[vote] = 0 if vote not in votes else votes[vote] + 1
        #     plurality = max(votes, key=votes.get)
        #     predictions.append(plurality)
        #     votes.clear()
        # return predictions

    
class Distance:
    def normalize(X):
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        return np.nan_to_num((X - mins) / (maxs - mins))
    
    def euclidean_dist(x1, x2):
        diff = x1-x2
        return np.sqrt(diff @ diff)
    
    def squared_euclidean_dist(x1, x2):
        diff = x1 - x2
        return diff @ diff

    def manhattan_dist(x1, x2):
        return sum(abs(x1-x2))
    
    def minkowski_dist(power_root, x1, x2):
        """power_root=1 is Manhattan distance
        power_root=2 is Euclidean distance
        Partial application needed for compatibility with predict.
        """
        diff = x1 - x2
        return np.power(sum(diff ** power_root), 1/power_root)
    
    def chebyshev_dist(x1, x2):
        return np.max(np.abs(x1-x2))
    
    def cosine_similarity(x1, x2):
        if sum(x1) == 0 or sum(x2) == 0:
            return 0 #not sure what to do here
        return -((x1 @ x2) / (np.sqrt(x1 @ x1)*np.sqrt(x2 @ x2))) #return the opposite of the cosine similarity to be compatible with sort by increasing value
    

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






