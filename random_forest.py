from Classifier import RandomForestClassifier
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from preprocessing import get_data

def main():
    argc = len(sys.argv)
    if argc < 5:
        print("Usage: python3 randomForest.py <TrainingSetFile.csv> <NumAttributes> <NumDataPoints> <NumTrees>")
        sys.exit()

    training_file = sys.argv[1]
    num_attributes = int(sys.argv[2])
    num_data_points = int(sys.argv[3])
    num_trees = int(sys.argv[4])
    
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
    