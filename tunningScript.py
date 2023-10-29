from Classifier import RandomForestClassifier
import os
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
    num_attributes_list = [2,3,4,5,6]  # Example values, adjust based on your dataset and preference
    num_trees_list = [100, 200]  # Example values, adjust as needed
    
        # Placeholder for the best model's metrics
    best_accuracy = 0
    best_params = {}
    
    argc = len(sys.argv)
    if argc < 3: #lol <3
        print("Usage: tunningScript.py <TrainingSetFile.csv> <NumDataPoints>")
        sys.exit()


    try:
        training_file = sys.argv[1]
        num_data_points = int(sys.argv[2])
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
                # After each model evaluation, plot the confusion matrix and save it
                class_names = np.unique(y)
                class_names.sort()
                precisions, recalls = precision_recall_by_class(matrix)
                f_measures = [f_measure(p, r) for p, r in zip(precisions, recalls)]
                
                # Check if the directory exists
                if not os.path.exists("output"):
                    # If not, create the directory
                    os.makedirs("output")
                
                plot_title = f"Confusion Matrix NumAttributes_{curr_num_attributes}_NumTrees_{curr_num_trees}"
                plot_confusion_matrix(matrix, class_names, precisions, recalls, f_measures, title=plot_title)
                plt.savefig(os.path.join("output", f"{plot_title}.png"))  # This saves the current confusion matrix plot

            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                

    print(f"Execution finished. Log file and confusion matrices have been saved. Best model parameters: {best_params} with accuracy: {best_accuracy}")

    logging.info(f"Best parameters: {best_params} with accuracy: {best_accuracy}")


if __name__ == "__main__":
    main()