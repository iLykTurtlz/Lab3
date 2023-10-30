from Classifier import KNNClassifier
from preprocessing import get_data, normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from validation import *



def main():
    valid_metrics = ("euclidean", "manhattan", "cosine", "chebyshev", "minkowski")
    argc = len(sys.argv)
    if argc not in (5,6):
        print(f"Usage: python3 knn.py <Dataset.csv> <min value of k> <max value of k> <distance - from {valid_metrics}> [value of p for Minkowski distance]")
        sys.exit()
    p = None
    try:
        training_file = sys.argv[1]
        min_k = int(sys.argv[2])
        max_k = int(sys.argv[3])
        distance = sys.argv[4]
        if distance not in valid_metrics:
            print(f"The distance metric must be from {valid_metrics}")
            sys.exit()
        if argc == 6:
            p = float(sys.argv[5])
    except:
        print("Couldn't read one or more input parameters.")
        sys.exit()

    # X=None
    # y=None
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
        X = normalize(X)    #does nothing if the data is all categorical
    except Exception as e:
        print("Could not preprocess the dataset")
        print(str(e))
        sys.exit()
    
    #print the parameters:
    print("python", end=" ")
    for arg in sys.argv:
        print(arg, end=" ")
    print("\n")



    # Prepare the results file
    knn = KNNClassifier(1, distance, p)
    
    nb_folds=10 

    matrices, accuracies, avg_accuracies = knn_cross_validation(knn, X, y, nb_folds, min_k, max_k)

    for i, matrix in enumerate(matrices):
        k = min_k + i
        print("Confusion matrix for k =", k)
        print(matrix)
        class_names = np.unique(y)
        class_names.sort()

        #print("Accuracy per crossval iteration:", accuracies)
        #print("Average accuracies:",avg_accuracies)
        print("Average accuracy:",avg_accuracies[i])
        print()
        # print("Accuracy for each cross-validation split:")
        # for acc in accuracies:
        #     print("\t"+str(accuracies[i]))
        
        precisions, recalls = precision_recall_by_class(matrix)
        f_measures = [f_measure(p,r) for p,r in zip(precisions, recalls)]



        row_names = ['Precision', 'Recall', 'F-measure']
        data = [precisions, recalls, f_measures]

        column_width = max(
            max(len(str(row[i])) for row in data) for i in range(len(class_names))
        )
        column_width = max(column_width, max(len(name) for name in row_names))
        column_width = max(column_width, 12)

        print(f"{' ' * column_width:<{column_width}}", end=' ')
        for name in class_names:
            print(f"{name:^{column_width}}", end=' ')
        print()

        for i, row in enumerate(data):
            print(f"{row_names[i]:<{column_width}}", end=' ')
            for item in row:
                print(f"{item:>{column_width}.6f}", end=' ')
            print()

        print("_"*80)
        print()

        #print("f-measures by class:",f_measures)

        
        # print("Class labels:",class_names)
        # print("Precision by class:",precisions)
        # print("Recalls by class:", recalls)


        # f_measures = [f_measure(p,r) for p,r in zip(precisions, recalls)]
        # print("f-measures by class:",f_measures)

        #plot_confusion_matrix(matrix, class_names, precisions, recalls, f_measures)
    plt.plot(range(min_k, max_k+1), avg_accuracies, linestyle='--', marker='o', color='r')
    plt.xlabel("k")
    plt.ylabel('accuracy')
    plt.title("Accuracy over ten-fold cross validation vs. k")
    plt.show()

        #print()

if __name__ == "__main__":
    main()
    