from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd

def entropy(D, class_label):
    n = D.shape[0]
    classes, counts = np.unique(D[class_label], return_counts=True)
    probabilities = counts / n
    probabilities = np.flatnonzero(probabilities) #otherwise log of 0 will be an issue
    if len(probabilities) == 1:
        return 0
    s = probabilities.sum()
    if s > 0.0001 or s < -0.0001:
        raise Exception("Not a probability distribution.")
    return - np.sum([p * math.log2(p) for p in probabilities])

class Leaf:
    pass

class CategoricalNode:
    pass

class NumericNode:
    pass


class DecisionTree(ABC):
    def __init__(self):
        self.root = None

    @abstractmethod
    def build(self, data, classes, attributes):
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


class CategoricalDecisionTree(DecisionTree):
    def __init__(self):
        super().__init__()





