from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
from collections import defaultdict




class Leaf:
    """Contains the classification label at the end of a tree branch.
    """
    def __init__(self, label):
        self.label = label

class CategoricalNode:
    """Splits on known values of the splitting attribute, which are the keys in the dictionary
    of children.  The corresponding values are child subtrees following the edges labeled
    with the keys.
    """
    def __init__(self, children, splitting_attr):
        self.children = children
        self.splitting_attr = splitting_attr

class NumericalNode:
    """Splits on an attribute at a certain splitting value.  If a data point has a value
    for the attribute less than or equal to the splitting value, it is categorized as 
    belonging to the left subtree.  Otherwise it is categorized as belonging to the right
    subtree.
    """
    def __init__(self, left, right, splitting_attr, splitting_value):
        self.left = left
        self.right = right
        self.splitting_attr = splitting_attr
        self.splitting_value = splitting_value

    



class DecisionTree(ABC):
    """The abstract base class of a decision tree.
    """
    def __init__(self):
        self.root = None

    @abstractmethod
    def fit(self, X, y, threshold=0.001): #X = data, y = labels
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
    
    @abstractmethod
    def predict(self, x):
        """
        returns a class label for the data point x
        """
        raise NotImplementedError("This method has not been implemented")
    
    def entropy(y):
        n = y.shape[0]
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n
        if len(probabilities) == 1:
            return 0
        zero = probabilities.sum() - 1
        if zero > 0.0001 or zero < -0.0001:
            raise Exception("Not a probability distribution.")
        return - np.sum([p * math.log2(p) for p in probabilities])

    def entropy_split(X, y, a):
        """
        Returns the entropy after a (hypothetical) split on the attribute a
        If a is real-valued or integral with more than 5 distinct values, 
        this function will find the best splitting threshold to create two child nodes.
        
        """
        

        

    def plurality(y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]


class CategoricalDecisionTree(DecisionTree):
    def __init__(self):
        super().__init__()

    def select_splitting_attribute(X, y, A, threshold, ratio=False):
        p_0 = DecisionTree.entropy(y)
        max_gain = -1 #information gain cannot be negative
        best = None
        for a in A:
            p_a = DecisionTree.entropy_split(X, y, a)
            gain_a = p_0 - p_a
            if ratio:
                gain_a = gain_a / DecisionTree.entropy(X[a])
            if gain_a > max_gain:
                max_gain = gain_a
                best = a 
        if max_gain > threshold:
            return best
        else:
            return None

    def C45(X: pd.core.frame.DataFrame, y: pd.core.series.Series, A, threshold, ratio=False):
        if y.unique().shape[0] == 1:
            return Leaf(y.iloc[0])
        elif len(A) == 0:
            return Leaf(DecisionTree.plurality(y))
        else:
            splitting_attr = CategoricalDecisionTree.select_splitting_attribute(X, y, A, threshold, ratio)
            if splitting_attr is None:
                return Leaf(DecisionTree.plurality(y))
            else:
                p = DecisionTree.plurality(y)
                children = defaultdict(lambda : Leaf(p))
                splitting_domain = np.unique(X[splitting_attr])
                for v in splitting_domain:
                    Xv = X[X[splitting_attr] == v]
                    yv = y[X[splitting_attr] == v]
                    if Xv.shape[0] != 0:  #this test is probably useless
                        Av = [a for a in A if a != splitting_attr] 
                        Tv = CategoricalDecisionTree.C45(Xv, yv, Av, threshold, ratio)
                        children[v] = Tv
                return CategoricalNode(children, splitting_attr)

    def fit(self, X, y, threshold):
        self.root = CategoricalDecisionTree.C45(X, y, X.columns, threshold)

    def predict(self, X):
        def tree_search(x, current):
            if isinstance(current, Leaf):
                return current.label
            elif isinstance(current, CategoricalNode):
                return tree_search(x, current.children[x[current.splitting_attr]])
            else:
                raise Exception("Tree error: a node that is neither CategoricalNode nor Leaf.")
        return [tree_search(x, self.root) for x in X]
        






