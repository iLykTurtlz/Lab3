import numpy as np
import pandas as pd

def k_folds(X : pd.core.frame.DataFrame, y, k):
    """If k==0 or k==1, returns the same X and y passed in.
    Otherwise returns a generator yielding (X_train, X_test), (y_train, y_test).
    If k==-1, generates leave one out splits.
    Otherwise generates k such sets, k-fold
    """
    if k==0 or k==1:
        print("No effect")
        return X,y
    elif k==-1:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        for i in range(len(X)):
            X_train = X.drop(X.index[i])
            X_test = X.iloc[[i]]
            y_train = y.drop(y.index[i])
            y_test = y.iloc[[i]]
            yield (X_train, X_test), (y_train, y_test)
    elif k > 1:
        shuffled_idx = np.random.permutation(X.index)
        shuffled_X = X.reindex(shuffled_idx)
        shuffled_y = y.reindex(shuffled_idx)
        X = shuffled_X.reset_index(drop=True)
        y = shuffled_y.reset_index(drop=True)
        size = X.shape[0] // k
        remainder = X.shape[0] % k
        Xs = [X.iloc[i*size:(i+1)*size] for i in range(k)]
        ys = [y.iloc[i*size:(i+1)*size] for i in range(k)]
        remainingX = X.tail(remainder)
        remainingy = y.tail(remainder)
        for i, ((_,row), (_,item)) in enumerate(zip(remainingX.iterrows(), remainingy.items())):
            Xs[i].loc[len(Xs[i])] = row
            ys[i].loc[len(ys[i])] = item
        #assert(sum(len(x) for x in Xs) == len(X) and sum(len(Y) for Y in ys) == len(y))
        fold_indices = [([i],[j for j in range(len(Xs)) if i!=j]) for i in range(len(Xs))]
        return (  ((pd.concat([Xs[j] for j in train]), Xs[test[0]]), (pd.concat([ys[j] for j in train]), ys[test[0]])) for test, train in fold_indices  )



def cross_validation(classifier, X, y, k, threshold, ratio=False):
    labels = np.unique(y)
    labels.sort()
    encoding = {label:number for number,label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros(shape=(n,n), dtype=int)
    accuracies = np.zeros(k)
    global_accuracy = 0
    total_predictions = 0
    for i, ((X_train, X_test), (y_train, y_test)) in enumerate(k_folds(X,y,k)):
        classifier.fit(X_train, y_train, threshold, ratio)
        predictions = classifier.predict(X_test)
        for y_true, y_pred in zip(y_test, predictions):
            matrix[encoding[y_true], encoding[y_pred]] += 1
            if y_true == y_pred:
                accuracies[i] += 1
                global_accuracy += 1
        accuracies[i] /= len(y_true)
        total_predictions += len(y_true)
    global_accuracy /= total_predictions
    assert(total_predictions == X.shape[0])
    avg_accuracy = sum(accuracies) / k
    return matrix, accuracies, global_accuracy, avg_accuracy

            
def precision_recall_by_class(matrix):
    precision_by_class = [matrix[i,i]/sum(matrix[:,i]) for i in range(matrix.shape[1])]
    recall_by_class = [matrix[i,i]/sum(matrix[i,:]) for i in range(matrix.shape[0])]
    return precision_by_class, recall_by_class

def overall_recall_precision(matrix, average="macro"):
    precisions, recalls = precision_recall_by_class(matrix)
    if average == "macro":
        return sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    elif average == "micro": #this will return the same number twice
        true_pos = np.diag(matrix)
        sum_rows = np.sum(matrix, axis=0)
        sum_cols = np.sum(matrix, axis=1)
        false_pos = np.array([x - true_pos[i] for i,x in enumerate(sum_rows)])
        false_neg = np.array([x - true_pos[i] for i,x in enumerate(sum_cols)])
        return sum(true_pos) / (sum(true_pos) + sum(false_pos)), sum(true_pos) / (sum(true_pos) + sum(false_neg))
    else:
        print("Invalid arg: average")
    
def f_measure(precision, recall):
    return 2*precision*recall / (precision + recall)

def pf(matrix, average="macro"): 
    true_pos = np.diag(matrix)
    true_neg = np.array([sum([true_pos[i] for i in range if i != j]) for j in range(len(true_pos))])
    sum_rows = np.sum(matrix, axis=0)
    false_pos = np.array([x - true_pos[i] for i,x in enumerate(sum_rows)])
    pf_scores = np.array([x / (x+y) for x,y in zip(false_pos, true_neg)])
    if average == "macro":
        return pf_scores, sum(pf_scores) / len(pf_scores)

# def confusion_matrix_str(matrix, X):







        





