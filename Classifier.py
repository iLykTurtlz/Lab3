from abc import ABC, abstractmethod
from InduceC45 import CategoricalDecisionTree
import sys
import pandas as pd
import json

class Classifier(ABC):
    """
    This is the abstract base class for classifiers.
    """
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
    
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("This method has not been implemented.")
    
    @abstractmethod
    def predict(self, x):
        raise NotImplementedError("This method has not been implemented.")
    
    @abstractmethod
    def read_json(self, filename):
        raise NotImplementedError("This method has not been implemented.")
    
    

class DecisionTreeClassifier(Classifier):
    """
    This classifier creates a DecisionTree instance when the fit method is called.  
    We may eventually have different types of decision trees.
    If the predict method is called before the fit method, or if asked to fit incompatible data, an error is raised.
    """
    def __init__(self, input_dimension, threshold, kind='categorical', ratio=False):
        super().__init__(input_dimension)
        self.threshold = threshold
        self.kind = kind
        self.tree = None

    def fit(self, X, y, threshold=0.01):
        if self.kind == 'categorical':
            self.tree = CategoricalDecisionTree().build(X, y, threshold)

    


class KNNClassifier(Classifier):
    """
    This classifier stores the data passed in when the fit method is called.
    A variety of distance measures will eventually be possible, but it is Euclidean by default.
    If the predict method is called before the fit method, or if asked to fit incompatible data, an error is raised.
    """
    def __init__(self, input_dimension, distance = "Euclidean"):
        super().__init__(input_dimension)
        self.distance = distance
        self.independent_variables = None
        self.dependent_variables = None
    
    def fit(self, X, y):
        if len(X) != len(y):
            raise Exception("The independent and dependent variable lists must be compatible.")
        self.X = X
        self.y = y


def main():
    argc = len(sys.argv)
    if  argc < 3:
        print("Usage: python classifier.py <csv file> <json file>")
        sys.exit()
        
    test_file = sys.argv[1]
    json_file = sys.argv[2]
    
    try:
        test_data = pd.read_csv(test_file)
        tree = CategoricalDecisionTree()
        with open(json_file, 'r') as f:
            tree.from_dict(json.load(f))
        
        
            
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    

    


if __name__ == "__main__":
    main()


