from Classifier import RandomForestClassifier
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from preprocessing import get_data
from validation import *

def main():
    argc = len(sys.argv)
    if argc < 5:
        print("Usage: python3 randomForest.py <TrainingSetFile.csv> <NumAttributes> <NumDataPoints> <NumTrees>")
        sys.exit()

    try:
        training_file = sys.argv[1]
        num_attributes = int(sys.argv[2])
        num_data_points = int(sys.argv[3])
        num_trees = int(sys.argv[4])
    except:
        print("Couldn't read one or more input parameters.")
        sys.exit()

    X=None
    y=None
    # Load the dataset
    try:
        dataset = pd.read_csv(training_file)
    except Exception as e:
        print(f"Error: Could not read train file: {training_file}")
        print(str(e))
        sys.exit()
        
     # Preprocess the dataset
    try:    
        X, y = get_data(dataset)
    except Exception as e:
        print("Could not process the dataset")
        print(str(e))
        sys.exit()
    
    # Prepare the results file
    results_file = open('results.csv', 'w') 

    forest = RandomForestClassifier(num_attributes, num_data_points, num_trees, threshold=0.01, ratio=True)
    
    k=10 #ten-fold
    matrix, accuracies, avg_accuracy = cross_validation(forest, X, y, k)
    print("Confusion matrix:")
    print(matrix)
    class_names = np.unique(y)
    class_names.sort()

    print("Accuracy per crossval iteration:", accuracies)
    print("Average accuracy:",avg_accuracy)
    
    precisions, recalls = precision_recall_by_class(matrix)

    print("Class labels:",class_names)
    print("Precision by class:",precisions)
    print("Recalls by class:", recalls)
    f_measures = [f_measure(p,r) for p,r in zip(precisions, recalls)]
    print("f-measures by class:",f_measures)

    plot_confusion_matrix(matrix, class_names, precisions, recalls, f_measures)

    print()

if __name__ == "__main__":
    main()