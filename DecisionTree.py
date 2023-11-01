from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from pandas.api.types import is_any_real_numeric_dtype

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

class NumericNode:
    """Splits on an attribute at a certain splitting value.  If a data point has a value
    for the attribute less than or equal to the splitting value, it is categorized as 
    belonging to the left subtree.  Otherwise it is categorized as belonging to the right
    subtree.
    """
    def __init__(self, left, right, splitting_attr, splitting_value, default: Leaf):
        self.left = left
        self.right = right
        self.splitting_attr = splitting_attr
        self.splitting_value = splitting_value
        self.default = default
    
    def to_dict(self):
        edges = [
            {'edge': {'value': '<= '+str(self.splitting_value), self.left.get_type(): self.left.to_dict()}},
            {'edge': {'value': '> '+str(self.splitting_value), self.right.get_type(): self.right.to_dict()}},
            {'edge': {'value': 'default', 'leaf': self.default.to_dict()}}
        ]
        return {"var": str(self.splitting_attr), "edges": edges}

    def get_type(self):
        return "node"
    
class DecisionTree(ABC):
    """The abstract base class of a decision tree.
    """
    def __init__(self):
        self.root = None

    @abstractmethod
    def fit(self, X, y, threshold=0.001, ratio=False): #X = data, y = labels
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
        if len(y) == 0:   #all null => 0 entropy
            return 0
        n = y.shape[0]
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / n
        if len(probabilities) == 1:
            return 0
        zero = probabilities.sum() - 1
        if zero > 0.0001 or zero < -0.0001:
            print(y)
            print(probabilities)
            raise Exception("Not a probability distribution.")
        return - np.sum([p * math.log2(p) for p in probabilities])

    def entropy_split(X, y, a):
        """
        Returns the entropy after a (hypothetical) split on the attribute a        
        """
        X_no_na = X.dropna()
        y_no_na = y.drop(index=y.index.difference(X_no_na.index))
        #assert(X.shape[0] == y.shape[0])
        n = X_no_na.shape[0]
        domain_a, counts = np.unique(X_no_na[a], return_counts=True)
        result = 0
        for value, count in zip(domain_a, counts):
            probability = count / n
            result += probability * DecisionTree.entropy(y_no_na[X_no_na[a] == value])
        return result

        

    def plurality(y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)], max(counts) / y.shape[0]


class CategoricalDecisionTree(DecisionTree):
    def __init__(self):
        super().__init__()

    def select_splitting_attribute(self, X, y, A, threshold, ratio):
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
                if (entropy_a := DecisionTree.entropy(X[a].dropna())) == 0:
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

    def C45(self, X: pd.core.frame.DataFrame, y: pd.core.series.Series, A, threshold, ratio):
        if y.unique().shape[0] == 1:
            return Leaf(y.iloc[0], 1.0)
        elif len(A) == 0:
            label, p = DecisionTree.plurality(y)
            return Leaf(label, p)
        else:
            splitting_attr = self.select_splitting_attribute(X, y, A, threshold, ratio)
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
            elif isinstance(current, NumericNode):
                if x[current.splitting_attr] <= current.splitting_value:
                    return tree_search(x, current.left)
                elif x[current.splitting_attr] > current.splitting_value:
                    return tree_search(x, current.right)
                else:
                    return tree_search(x, current.default)
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
                first_value = tree_dict['edges'][0]['edge']['value']
                #print(first_value)
                
                if str(first_value).startswith('<') or str(first_value).startswith('>'):
                    #assert(len(tree_dict['edges']) == 2)
                    splitting_value = float(first_value.split()[-1])
                    edge1 = tree_dict['edges'][0]
                    edge2 = tree_dict['edges'][1]
                    left = build(edge1['edge']['node']) if 'node' in edge1['edge'] else build(edge1['edge']['leaf'])
                    right= build(edge2['edge']['node']) if 'node' in edge2['edge'] else build(edge2['edge']['leaf'])
                    return NumericNode(left, right, attr, splitting_value, build(tree_dict['edges'][-1]['edge']['leaf']))
                else:
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


class CompleteDecisionTree(CategoricalDecisionTree):
    def __init__(self):
        super().__init__()

    def entropy_split(self, X, y, a, alpha):
        """
        """
        X_below, y_below, X_above, y_above = X[X[a] <= alpha], y[X[a] <= alpha], X[X[a] > alpha], y[X[a] > alpha]
        assert(X_below.shape[0] == y_below.shape[0])
        n1 = X_below.shape[0]
        n2 = X_above.shape[0]
        n = X.shape[0]
        entropy_below = DecisionTree.entropy(y_below)
        entropy_above = DecisionTree.entropy(y_above)
        return n1/n*entropy_below + n2/n*entropy_above

    def find_best_split(self, a, X, y, p_0):
        X_no_na = X.dropna(subset=[a])
        y_no_na = y.drop(index=y.index.difference(X_no_na.index))
        sorted_xa = np.unique(np.sort(X_no_na[a])) #need counts?
        potential_splits = [(sorted_xa[i] + sorted_xa[i+1])/2 for i in range(len(sorted_xa)-1)]
        best_alpha = None
        best_gain = -1
        for alpha in potential_splits:
            after_split = self.entropy_split(X_no_na, y_no_na, a, alpha)
            info_gain = p_0 - after_split
            if info_gain > best_gain:
                best_gain = info_gain
                best_alpha = alpha
        return best_alpha, best_gain

    def select_splitting_attribute(self, X, y, A, threshold, ratio):
        p_0 = DecisionTree.entropy(y)
        max_gain = -1 #information gain cannot be negative
        best = None #attribute
        is_numeric = False
        alpha = None #splitting threshold if best is_numeric
        for a in A:
            if len(X[a]) == 0:
                #print(a)
                continue
            if is_any_real_numeric_dtype(X[a]):
                best_alpha, info_gain = self.find_best_split(a, X, y, p_0)
                if info_gain > max_gain:
                    max_gain = info_gain
                    best = a 
                    is_numeric = True 
                    alpha = best_alpha 
            else:
                p_a = DecisionTree.entropy_split(X, y, a)
                gain_a = p_0 - p_a
                if ratio:
                    if (entropy_a := DecisionTree.entropy(X[a].dropna())) == 0:
                        continue
                    gain_a = gain_a / entropy_a
                if gain_a > max_gain:
                    max_gain = gain_a
                    best = a 
                    is_numeric = False 
        if max_gain > threshold:
            return best, is_numeric, alpha 
        else:
            return None, None, None
        
    def C45(self, X: pd.core.frame.DataFrame, y: pd.core.series.Series, A, threshold, ratio):
        if y.unique().shape[0] == 1:
            return Leaf(y.iloc[0], 1.0)
        elif len(A) == 0:
            label, p = DecisionTree.plurality(y)
            return Leaf(label, p)
        else:
            splitting_attr, is_numeric, alpha = self.select_splitting_attribute(X, y, A, threshold, ratio)
    
            label, p = DecisionTree.plurality(y)
            if splitting_attr is None:
                return Leaf(label, p)
            else:
                X = X.dropna(subset=[splitting_attr])
                y = y.drop(index=y.index.difference(X.index))
                if not is_numeric:
                    children = defaultdict(lambda : Leaf(label, p))
                    #splitting_values = X[splitting_attr]
                    #splitting_values = splitting_values.dropna()    #no null values will propagate down the tree
                    splitting_domain = np.unique(X[splitting_attr])
                   
                    for v in splitting_domain:
                        Xv = X[X[splitting_attr] == v]
                        yv = y[X[splitting_attr] == v]
                        if Xv.shape[0] != 0:  #this test is probably useless
                            Av = [a for a in A if a != splitting_attr] 
                            Tv = self.C45(Xv, yv, Av, threshold, ratio)
                            children[v] = Tv
                    return CategoricalNode(children, splitting_attr)
                else:
                    Xleft, Xright = X[X[splitting_attr] <= alpha], X[X[splitting_attr] > alpha]
                    yleft, yright = y[X[splitting_attr] <= alpha], y[X[splitting_attr] > alpha]
                    left = self.C45(Xleft, yleft, A, threshold, ratio)
                    right = self.C45(Xright, yright, A, threshold, ratio)
                    return NumericNode(left, right, splitting_attr, alpha, Leaf(label, p))
                
    def fit(self, X, y, threshold=0.01, ratio=False):
        self.root = self.C45(X, y, X.columns, threshold, ratio)
   


        

