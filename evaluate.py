import pandas as pd
from validation import *
from Classifier import DecisionTreeClassifier
import sys

def main():
    argc = len(sys.argv)
        
    if not 3 <= argc <= 4:
        print("python3 evaluate.py <CSVFile> [<restrictionsFile>] <numberOfFolds>")
        sys.exit()
        
    training = sys.argv[1]
    restrictions = sys.argv[2] if argc == 4 else None
    try:
        k = int(sys.argv[3]) if argc == 4 else int(sys.argv[2])
    except:
        print("The last argument must be a strictly positive integer.")
    
    try:
        data = pd.read_csv(training)
    except:
        print(f"Error: Could not read train file: {training}")
        

    
    if restrictions:
        try:
            with open(restrictions, 'r') as f:
                restrictions =  f.read().strip().split(",")
                restrictions = [int(i) for i in restrictions]
                
            if len(restrictions) != len(data.columns) - 1:
                print(f"Error: {sys.argv[2]} does not have the correct number of values, i.e. the number of columns without the class.")
                sys.exit()
        except:
            print("Restrictions file unreadable")
            sys.exit()

    X=None
    y=None
    try:
        class_col = data.iloc[1,0]
        data = data.drop(index=[0,1])
        X = data.loc[:,data.columns != class_col]
        y = data[class_col]
    except:
        print("Could not determine and/or separate category variable.")
        sys.exit()

    if restrictions:
        drop_cols = [col for col, keep in zip(X, restrictions) if not keep]
        X = X.drop(columns=drop_cols)

    c = DecisionTreeClassifier("categorical")
    c.fit(X, y, threshold=0.01, ratio=True)
    matrix, accuracies, avg_accuracy = cross_validation(c, X, y, k, threshold=0.01, ratio=True)
    print("Confusion matrix:")
    print(matrix)
    class_names = np.unique(y)
    class_names.sort()
    
    print("Accuracy per crossval iteration:", accuracies)
    print("Average accuracy:",avg_accuracy)
    
    precisions, recalls = precision_recall_by_class(matrix)
    print("Precision by class",precisions)
    print("Recalls by class:", recalls)
    f_measures = [f_measure(p,r) for p,r in zip(precisions, recalls)]
    print("f-measures by class:",f_measures)

    plot_confusion_matrix(matrix, class_names, precisions, recalls, f_measures)
    # pfs, avg_pf_score = pf(matrix)
    # print("pf-scores by class", pfs)
    # print("Average pf-score:",avg_pf_score)



if __name__ == "__main__":
    main()
