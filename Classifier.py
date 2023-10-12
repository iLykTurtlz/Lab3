from abc import ABC, abstractmethod


class Classifier(ABC):
    """
    This is the abstract base class for classifiers.
    """
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
    
    @abstractmethod
    def fit(self, independent_variables, dependent_variables):
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
    def __init__(self, input_dimension, min_info_gain):
        super().__init__(input_dimension)
        self.min_info_gain = min_info_gain
        self.tree = None


    


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
    
    def fit(self, independent_variables, dependent_variables):
        if len(independent_variables) != len(dependent_variables):
            raise Exception("The independent and dependent variable lists must be compatible.")
        self.independent_variables = independent_variables
        self.dependent_variables = dependent_variables
    

