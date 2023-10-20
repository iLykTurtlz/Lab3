from abc import ABC, abstractmethod
from DecisionTree import CategoricalDecisionTree
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

class Classifier(ABC):
    """
    This is the abstract base class for classifiers.
    """
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("This method has not been implemented.")
    
    @abstractmethod
    def predict(self, x):
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
        
    def calculate_confusion_matrix(self, y_true, y_pred):
        """
        Manually calculate the confusion matrix.

        :param y_true: List of true labels
        :param y_pred: List of predicted labels
        """
        # Unique classes in the true labels
        classes = np.unique(y_true)
        num_classes = len(classes)

        # Initialize the confusion matrix as a 2D array of zeros
        confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

        # Create a mapping from class labels to indices
        class_to_index = dict((current_class, index) for index, current_class in enumerate(classes))

        # Update the confusion matrix
        for actual, predicted in zip(y_true, y_pred):
            actual_index = class_to_index[actual]
            predicted_index = class_to_index[predicted]
            confusion_mat[actual_index, predicted_index] += 1

        return confusion_mat
    
    def plot_confusion_matrix(self, confusion_matrix, class_names):
        fig, ax = plt.subplots()
        cax = ax.matshow(confusion_matrix, cmap='Blues')
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, str(confusion_matrix[i, j]), va='center', ha='center')
        
        # Annotate errors of commission and omission
        for i in range(len(class_names)):
            error_of_commission = sum(confusion_matrix[i, :]) - confusion_matrix[i, i]
            error_of_omission = sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
            
            ax.text(len(class_names), i, f'{error_of_omission}', va='center', ha='center', color='red')
            ax.text(i, len(class_names), f'{error_of_commission}', va='center', ha='center', color='red')

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.xaxis.set_ticks_position('bottom')
        
        plt.tight_layout()
        plt.show()
    

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
    if sys.argc != 2:
        print("Usage: python classifier.py <csv file>")
        quit()
    
    

    


if __name__ == "__main__":
    main()


