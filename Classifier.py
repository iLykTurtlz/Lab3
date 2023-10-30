from abc import ABC, abstractmethod
from DecisionTree import CategoricalDecisionTree, CompleteDecisionTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from pandas.core.dtypes.common import is_numeric_dtype
from icecream import ic


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
        #assert(X.shape[0] == y.shape[0])
        # #X = X.reindex(drop=True) #for safety.  They must have the SAME index
        # #y = y.reindex(drop=True)
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
            predictions,_ = tree.predict(X)
            prediction_list.append(predictions)
        result = []
        for i in range(len(X)):
            votes = np.array([pred[i] for pred in prediction_list])
            values, counts = np.unique(votes, return_counts=True)
            result.append(values[np.argmax(counts)])
        return result, None

    
class Distance:
    # def separate_discrete_continuous(D):
    #     discrete = [col for col in D.columns if not is_numeric_dtype(D[col])]
    #     continuous = [col for col in D.columns if is_numeric_dtype(D[col])]
    #     return D[discrete], D[continuous]
    
    def dice(x1, x2):
        """Hypothesis: x1 and x2 are categorical only"""
        return 1 - (2 * len(x1[x1==x2]) / (len(x1) + len(x2)))
    
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
        diff = np.abs(x1 - x2)
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
    def __init__(self, k, distance="euclidean", power_root=None):
        super().__init__()
        self.k = k
        self.matrix = None
        if distance == "euclidean":
            self.dist=Distance.euclidean_dist
        elif distance == "manhattan":
            self.dist=Distance.manhattan_dist
        elif distance == "cosine":
            self.dist = Distance.cosine_similarity
        elif distance == "chebyshev":
            self.dist = Distance.chebyshev_dist
        elif distance == "minkowski":
            if power_root is None:
                raise Exception("minkowski distance requires a power_root argument.")
            self.dist = partial(Distance.minkowski_dist, power_root)
        else:
            raise ValueError
        
    def fit(self, X, y):
        if len(X) != len(y):
            raise Exception("The independent and dependent variable lists must be compatible.")
        self.data = X.reset_index(drop=True)
        self.labels = y.reset_index(drop=True)

    def most_frequent(self, row):
            values, counts = np.unique(row, return_counts=True)
            return values[np.argmax(counts)]

    def predict(self, X):
        if self.data is None or self.labels is None:
            raise NotFittedError(f"{type(self).__name__}.fit(X,y) must be called before this method.")
        if X.shape[1] != self.data.shape[1]:
            raise IncompatibleDimensionsError(f"This {type(self).__name__} can only make predictions on data points of shape {self.data[0].shape}")
        
        predictions = np.zeros(len(X), dtype=self.labels.dtype)
        self.matrix = []
        if X.select_dtypes(include='object').shape[1] > 0: #if there are categorical attributes
            cat_dist = Distance.dice
            cont_dist = self.dist
            catX = X.select_dtypes(include="object") 
            contX = X.select_dtypes(include=np.number)
            cat_data = self.data.select_dtypes(include='object')
            cont_data = self.data.select_dtypes(include=np.number)
            #ic(cat_data)
            #ic(cont_data.to_numpy())
            if catX.shape[1] != cat_data.shape[1] or contX.shape[1] != cont_data.shape[1]:
                raise IncompatibleDimensionsError(f"This {type(self).__name__} can only make predictions on data points with {cat_data.shape[1]} categorical attributes and {cont_data.shape[1]} continuous attributes.")
            nb_cat = cat_data.shape[1]
            nb_cont = cont_data.shape[1]
            nb_tot = self.data.shape[1]
            distances = np.zeros(len(self.data))
            for j, (x_cat, x_cont) in enumerate(zip(catX.to_numpy(), contX.to_numpy())):
                for i, (cat, cont) in enumerate(zip(cat_data.to_numpy(), cont_data.to_numpy())):
                    distances[i] = (nb_cat / nb_tot) * cat_dist(x_cat, cat) + (nb_cont / nb_tot) * cont_dist(x_cont, cont)
                #ic(distances)
                sorted_idx = distances.argsort()
                #ic(sorted_idx)
                self.matrix.append(sorted_idx)
                #ic(self.labels)
                nearest_neighbors = self.labels[sorted_idx]#[:self.k]
                #ic(nearest_neighbors)
                nearest_neighbors = nearest_neighbors.head(self.k)
                predictions[j] = self.most_frequent(nearest_neighbors)
            self.matrix = np.asarray(self.matrix)
        else:
            for i, x in enumerate(X.to_numpy()):
                sorted_idx = np.apply_along_axis(partial(self.dist, x), axis=1, arr=self.data.to_numpy()).argsort()
                self.matrix.append(sorted_idx)
                nearest_neighbors = self.labels[sorted_idx].head(self.k)
                predictions[i] = self.most_frequent(nearest_neighbors)
            self.matrix = np.asarray(self.matrix)
        return predictions, None

    def predict_for_krange(self, min_k, max_k):
        if self.matrix is None:
            raise NotFittedError(f"{type(self).__name__}.fit(X,y) must be called before this method.")
        if max_k > self.matrix.shape[1]:
            max_k = self.matrix.shape[1]
        for k in range(min_k, max_k + 1):
            nearest_neighbors = np.array([self.labels[row] for row in self.matrix[:,:k]])
            yield k, [self.most_frequent(ns) for ns in nearest_neighbors] #np.apply_along_axis(self.most_frequent, axis=1, arr=nearest_neighbors) #apply_along_axis truncates strings!  No kwarg to prevent this!
    

        





