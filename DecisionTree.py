from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import json

class TreeReadError(Exception):
    def __init__(self, message):
        super().__init__(message)

class Leaf:
    """Contains the classification label at the end of a tree branch.
    p is the probability of the plurality class
    """
    def __init__(self, label, p):
        self.label = label
        self.p = p

    def to_dict(self):
        return {"decision": str(self.label), "p" : self.p}
    
    def get_type(self):
        return "leaf"

class CategoricalNode:
    """Splits on known values of the splitting attribute, which are the keys in the dictionary
    of children.  The corresponding values are child subtrees following the edges labeled
    with the keys.
    """
    def __init__(self, children, splitting_attr):
        self.children = children
        self.splitting_attr = splitting_attr

    def to_dict(self):
        edges = [{"edge": {"value": k, v.get_type(): v.to_dict()}} for k,v in self.children.items()]
        edges.append({"edge" : {"value": "default", "leaf": self.children.default_factory().to_dict()}})
        return {"var": str(self.splitting_attr), "edges": edges}
    
    def get_type(self):
        return "node"

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
        return labels[np.argmax(counts)], max(counts) / y.shape[0]


class CategoricalDecisionTree(DecisionTree):
    def __init__(self):
        super().__init__()

    def select_splitting_attribute(X, y, A, threshold, ratio):
        p_0 = DecisionTree.entropy(y)
        max_gain = -1 #information gain cannot be negative
        best = None
        for a in A:
            if len(X[a]) == 0:
                #print(a)
                continue
            p_a = DecisionTree.entropy_split(X, y, a)
            gain_a = p_0 - p_a
            if ratio:
                #print("a=",a,"gain_a:",gain_a,"; Entropy of X[a]", DecisionTree.entropy(X[a]), "len(X[a])=",len(X[a]))
                if (entropy_a := DecisionTree.entropy(X[a])) == 0:
                    #print(np.unique(X[a], return_counts=True))
                    continue
                gain_a = gain_a / entropy_a
            if gain_a > max_gain:
                max_gain = gain_a
                best = a 
        if max_gain > threshold:
            return best
        else:
            return None

    def C45(X: pd.core.frame.DataFrame, y: pd.core.series.Series, A, threshold, ratio):
        if y.unique().shape[0] == 1:
            return Leaf(y.iloc[0], 1.0)
        elif len(A) == 0:
            label, p = DecisionTree.plurality(y)
            return Leaf(label, p)
        else:
            splitting_attr = CategoricalDecisionTree.select_splitting_attribute(X, y, A, threshold, ratio)
            if splitting_attr is None:
                label, p = DecisionTree.plurality(y)
                return Leaf(label, p)
            else:
                label, p = DecisionTree.plurality(y)
                children = defaultdict(lambda : Leaf(label, p))
                splitting_domain = np.unique(X[splitting_attr])
                for v in splitting_domain:
                    Xv = X[X[splitting_attr] == v]
                    yv = y[X[splitting_attr] == v]
                    if Xv.shape[0] != 0:  #this test is probably useless
                        Av = [a for a in A if a != splitting_attr] 
                        Tv = CategoricalDecisionTree.C45(Xv, yv, Av, threshold, ratio)
                        children[v] = Tv
                return CategoricalNode(children, splitting_attr)

    def fit(self, X, y, threshold, ratio=False):
        self.root = CategoricalDecisionTree.C45(X, y, X.columns, threshold, ratio)

    def predict(self, X):
        def tree_search(x, current):
            if isinstance(current, Leaf):
                return current.label, current.p
            elif isinstance(current, CategoricalNode):
                return tree_search(x, current.children[x[current.splitting_attr]])
            else:
                print("Type of node=",type(current))
                raise Exception("Tree error: a node that is neither CategoricalNode nor Leaf.")
        #result = [tree_search(X.iloc[i], self.root) for i in range(X.shape[0])] #this result is a list of tuples
        
        # result = [],[]  #this result is a tuple of lists: labels, confidences
        # for i in range(X.shape[0]):
        #     label, confidence = tree_search(X.iloc[i], self.root)
        #     result[0].append(label)
        #     result[1].append(confidence)

        #is this faster?
        result = [tree_search(row, self.root) for _,row in X.iterrows()]
        return [x for x,_ in result],[y for _,y in result]

    def to_dict(self):
        return self.root.to_dict()
    
    def from_dict(self, tree_dict):
        def build(tree_dict):
            if "decision" in tree_dict:
                return Leaf(tree_dict['decision'], tree_dict['p'])
            elif "var" in tree_dict:
                attr = tree_dict["var"]
                #edges = [(d['edge']['value'], build(d['edge']['node'])) for d in tree_dict['edges'][:-1]]
                edges = []
                for d in tree_dict['edges'][:-1]: #the last edge is the default edge, which should become a function returning a Leaf
                    if 'node' in d['edge']:
                        edges.append((d['edge']['value'], build(d['edge']['node'])))
                    elif 'leaf' in d['edge']:
                        edges.append((d['edge']['value'], build(d['edge']['leaf'])))
                    else:
                        raise TreeReadError("Found an edge that does not lead to an internal node or leaf.")
                children = defaultdict(lambda : build(tree_dict['edges'][-1]['edge']['leaf']), edges)
                return CategoricalNode(children, attr)
            else:
                raise TreeReadError("Found an improperly formatted internal node or leaf.")
        self.root = build(tree_dict)

    def to_json(self, data_file: str, write_file: str = None, return_str: bool = False):
        json_out = {
        "dataset" : data_file,
        "node": self.to_dict()
        }
        if write_file:
            with open(write_file, "w") as wf:
                json.dump(json_out, wf, indent = 2)
        if return_str:
            return str(json_out)
        
    def from_json(self, file):
        dico = None
        with open(file, "r", encoding="utf-8") as f:
            dico = json.load(f)
        self.from_dict(dico['node'])


        

