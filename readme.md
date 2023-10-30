#Decision Trees, Random Forests, KNN

Paul Jarski     pjarski@calpoly.edu
Joe Dewar       jdewar@calpoly.edu

Note: incoming csv data files are expected to be in the same format as the originals: with a column giving the domain size for each class and a row dedicated to the UNIQUE label of the class.  Our programs will remove these two rows after extracting their information.


Executable files:

induceC45.py        usage: python3 InduceC45 <TrainingSetFile.csv> [<restrictionsFile>]
                    result: creates and writes a json file, tree.json, representing a C4.5 decision tree

classify.py         usage: python3 classify.py <CSVFile> <JSONFile>
                    result: validation metrics for a C4.5 tree created from the JSON file.  

evaluate.py         usage: python3 evaluate.py <CSVFile> [<restrictionsFile>] <numberOfFolds or -1 for leave one out>
                    result: a confusion matrix and other metrics for cross validation of a C4.5 decision tree model.

random_forest.py    usage: python3 random_forest.py <TrainingSetFile.csv> <NumAttributes> <NumDataPoints> <NumTrees>
                    result: 10-fold cross validation metrics for a random forest using the given hyperparameters

knn.py              usage:  python3 knn.py <Dataset.csv> <min value of k> <max value of k> <distance - from {valid_metrics}> [value of p for Minkowski distance]
                    valid metrics are {"euclidean", "manhattan", "cosine", "chebyshev", "minkowski"}
                    result: 10-fold cross validation for a knn classifier for values of k from the min to max values given, inclusive.  Also a graph of overall accuracy for each value k.

Dependencies:

Classifier.py - interface and instantiable classifiers
DecisionTree.py - code for the C4.5 algorithm
preprocessing.py - functions to extract data points and labels from a DataFrame
validation.py - functions for k folds and cross-validation.

Output files:

KNN_output - includes accuracy graphs and output logs for each cross-validation run.  Our report uses a subset of these to make its arguments.


