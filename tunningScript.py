from Classifier import RandomForestClassifier
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from preprocessing import get_data
from validation import *
import logging
from datetime import datetime

def main():
    # Initialize logging
    logging.basicConfig(filename=f'randomForest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Placeholder for the best model's metrics
    best_accuracy = 0
    best_params = {}

    # Starting parameters for tuning
    num_attributes_list = [5, 10, 15]  # Example values, adjust based on your dataset and preference
    num_trees_list = [100, 200]  # Example values, adjust as needed
    
        # Placeholder for the best model's metrics
    best_accuracy = 0
    best_params = {}

    # Starting parameters for tuning
    num_attributes_list = [5, 10, 15]  # Example values, adjust based on your dataset and preference
    num_trees_list = [100, 200]  # Example values, adjust as needed
    
    argc = len(sys.argv)
    if argc < 5:
        print("Usage: python3 randomForest.py <TrainingSetFile.csv> <NumAttributes> <NumDataPoints> <NumTrees>")
        sys.exit()


    try:
        training_file = sys.argv[1]
        num_attributes = int(sys.argv[2])
        num_data_points = int(sys.argv[3])
        num_trees = int(sys.argv[4])
    except ValueError:
        print("Couldn't read one or more input parameters.")
        sys.exit()
        
    X = None
    y = None

    # Load and preprocess the dataset
    try:
        dataset = pd.read_csv(training_file)
        X, y = get_data(dataset)
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        sys.exit()



    for curr_num_attributes in num_attributes_list:
        for curr_num_trees in num_trees_list:
            logging.info(f"Evaluating model with NumAttributes: {curr_num_attributes}, NumTrees: {curr_num_trees}")

            try:
                forest = RandomForestClassifier(curr_num_attributes, num_data_points, curr_num_trees, threshold=1.0, ratio=True)
                matrix, accuracies, avg_accuracy = cross_validation(forest, X, y, k=10, threshold=1.0, ratio=True)

                logging.info(f"Confusion matrix:\n{matrix}")
                logging.info(f"Accuracy per iteration: {accuracies}")
                logging.info(f"Average accuracy: {avg_accuracy}")
                
                # Compare and save the best model's parameters
                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_params = {'NumAttributes': curr_num_attributes, 'NumTrees': curr_num_trees}

                logging.info(f"Average accuracy: {avg_accuracy}")

            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")

    logging.info(f"Best parameters: {best_params} with accuracy: {best_accuracy}")


