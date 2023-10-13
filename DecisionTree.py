from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd



class Leaf:
    def __init__(self, label):
        self.label = label

class CategoricalNode:
    pass


class DecisionTree(ABC):
    def __init__(self):
        self.root = None

    @abstractmethod
    def build(self, X, y, threshold=0.001): #X = data, y = labels
        """
        Constructs a tree and assigns the root to self.root
        """
        raise NotImplementedError("This method has not been implemented.")
    
    @abstractmethod
    def save(self, filename):
        """
        Writes a JSON file with the given name.
        """
        raise NotImplementedError("This method has not been implemented")
    
    def entropy(y):
        n = y.shape[0]
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n
        if len(probabilities) == 1:
            return 0
        zero = probabilities.sum() - 1
        if zero > 0.0001 or zero < -0.0001:                     #replace magic numbers
            raise Exception("Not a probability distribution.")
        return - sum([p * math.log2(p) for p in probabilities])

    def entropy_split(X, y, a):
        """
        Returns the entropy after a (hypothetical) split on the attribute a
        If a is real-valued or integral with more than 5 distinct values, 
        this function will find the best splitting threshold to create two child nodes.
        """
        assert(X.shape[0] == y.shape[0])
        n = X.shape[0]
        domain_a, counts = np.unique(X[a], return_counts=True)
        result = 0
        for value, count in zip(domain_a, counts):
            probability = count / n
            result += probability * DecisionTree.entropy(y[X[a] == value])
        return result

    def plurality(y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]
    
    def information_gain(X, y, a):
        pass

    def information_gain_ratio(X, y, a):
        pass


class CategoricalDecisionTree(DecisionTree):
    def __init__(self):
        super().__init__()

    def select_splitting_attribute(X, y, A, threshold):
        p_0 = DecisionTree.entropy(y)
        max_gain_ratio = -1 #information gain cannot be negative
        best = None
        for a in A:
            p_a = DecisionTree.entropy_split(X, y, a)
            gain_a = p_0 - p_a
            gain_ratio_a = gain_a / DecisionTree.entropy(X[a])
            if gain_ratio_a > max_gain_ratio:
                max_gain_ratio = gain_ratio_a
                best = a 
        if max_gain_ratio > threshold:
            return best
        else:
            return None

    def C45(X: pd.core.frame.DataFrame, y: pd.core.series.Series, A, threshold):
        if y.unique().shape[0] == 1:
            return Leaf(y.iloc[0])
        elif len(A) == 0:
            return Leaf(DecisionTree.plurality(y))
        else:
            pass

    def build(self, X, y, threshold = 0.001):
        self.root = CategoricalDecisionTree.C45(X, y, X.columns, threshold)

        
        






