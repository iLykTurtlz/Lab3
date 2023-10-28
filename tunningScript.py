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

    for curr_num_attributes in num_attributes_list:
        for curr_num_trees in num_trees_list:
            logging.info(f"Evaluating model with NumAttributes: {curr_num_attributes}, NumTrees: {curr_num_trees}")

            try:
                # Initialize and evaluate the model using current parameters
                forest = RandomForestClassifier(curr_num_attributes, num_data_points, curr_num_trees, threshold=0.01, ratio=True)
                matrix, accuracies, avg_accuracy = cross_validation(forest, X, y, k, threshold=0.01, ratio=True, write_file=None)  # Assuming this function returns the average accuracy

                # Compare and save the best model's parameters
                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_params = {'NumAttributes': curr_num_attributes, 'NumTrees': curr_num_trees}

                logging.info(f"Average accuracy: {avg_accuracy}")

            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")

    logging.info(f"Best parameters: {best_params} with accuracy: {best_accuracy}")


