import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def k_folds(X : pd.core.frame.DataFrame, y, k):
    """If k==0 or k==1, returns the same X and y passed in.
    Otherwise returns a generator yielding (X_train, X_test), (y_train, y_test) for k iterations.
    If k==-1, generates leave one out splits.
    Hypothesis: indices are contiguous and start at 0, X and y have the same indices
    """
    if k==0 or k==1:
        print("No effect")
        return (X,None),(y,None)
    elif k==-1:
        # X = X.reset_index(drop=True)
        # y = y.reset_index(drop=True)
        for i in X.index:
            X_train = X.drop(X.index[i])
            X_test = X.iloc[[i]]
            y_train = y.drop(y.index[i])
            y_test = y.iloc[[i]]
            yield (X_train, X_test), (y_train, y_test)
    elif k > 1:
        #assert(X.index.equals(y.index))
        indices = X.index.values
        np.random.shuffle(indices)
        #shuffled_idx = np.random.permutation(X.index)
        #shuffled_X = X.reindex(shuffled_idx)
        #shuffled_y = y.reindex(shuffled_idx)
        # X = shuffled_X.reset_index(drop=True)
        # y = shuffled_y.reset_index(drop=True)
        size = X.shape[0] // k
        remainder = X.shape[0] % k
        sizes = [size for _ in range(k)]
        for i in range(remainder):
            sizes[i] += 1
        curr = 0
        startstop = []
        for s in sizes:
            startstop.append((curr, curr+s))
            curr += s
        #assert(curr == X.shape[0])
        slice_indices = [indices[start:stop] for start, stop in startstop]
        Xs = [X.loc[idx_list] for idx_list in slice_indices]
        ys = [y.loc[idx_list] for idx_list in slice_indices]
        # Xs = [X.iloc[start:stop] for start, stop in startstop]
        # ys = [y.iloc[start:stop] for start, stop in startstop]
        #assert(sum(len(x) for x in Xs) == len(X) and sum(len(Y) for Y in ys) == len(y))
        fold_indices = [(i,[j for j in range(len(Xs)) if i!=j]) for i in range(len(Xs))]
        for test, train in fold_indices:
            X_train = pd.concat(Xs[i] for i in train)
            X_test = Xs[test]
            y_train = pd.concat(ys[i] for i in train)
            y_test = ys[test]
            yield (X_train, X_test), (y_train, y_test)


def cross_validation(classifier, X, y, k, threshold, ratio=False, write_file=None):
    if write_file:
        test_result = np.zeros(X.shape[0], dtype=y.dtype)
    labels = np.unique(y)
    labels.sort()
    encoding = {label:number for number,label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros(shape=(n,n), dtype=int)
    if k in (0,1):
        print("Nothing to validate if you take the entire dataset")
        return None, None, None
    elif k > 1:
        accuracies = np.zeros(k)
    elif k == -1:
        accuracies = np.zeros(X.shape[0])
    else:
        print("Invalid argument: k={k}")
        return None, None, None
    for i, ((X_train, X_test), (y_train, y_test)) in enumerate(k_folds(X,y,k)):
        #print("len train",len(X_train), "; len test", len(X_test))
        classifier.fit(X_train, y_train)
        predictions, _ = classifier.predict(X_test)
        if write_file:
            for j, prediction in zip(X_test.index, predictions):
                test_result[j] = prediction
        for y_true, y_pred in zip(y_test, predictions):
            matrix[encoding[y_true], encoding[y_pred]] += 1
            if y_true == y_pred:
                accuracies[i] += 1
        #print(y_true)
        accuracies[i] /= len(y_test)
    if write_file:
        results_series = pd.Series(test_result)
        frame_dict = {'True value': y, 'Predicted value': results_series}
        results_df = pd.DataFrame(frame_dict)
        results_df.to_csv(path_or_buf=write_file, index=False)

    avg_accuracy = sum(accuracies) / k
    return matrix, accuracies, avg_accuracy


def knn_cross_validation(knn_classifier, X, y, nb_folds, min_k, max_k):
    """Runs cross-validation and returns accuracy and confusion matrix results for k in range [min_k, max_k]."""
    labels = np.unique(y)
    labels.sort()
    encoding = {label:number for number,label in enumerate(labels)}
    n = len(labels)
    #matrix = np.zeros(shape=(n,n), dtype=int)
    if nb_folds in (0,1):
        print("Nothing to validate if you take the entire dataset")
        return None, None, None
    elif nb_folds > 1:
        accuracies = np.zeros(shape=(nb_folds, max_k - min_k + 1)) #one row for each fold, one column for each value of k
    elif nb_folds == -1:
        accuracies = np.zeros(shape=(X.shape[0], max_k - min_k + 1))
    else:
        print("Invalid argument: k={k}")
        return None, None, None
    matrices = [np.zeros(shape=(n,n), dtype=int) for _ in range(max_k - min_k + 1)]
    for i, ((X_train, X_test), (y_train, y_test)) in enumerate(k_folds(X,y,nb_folds)):
        #print("len train",len(X_train), "; len test", len(X_test))
        knn_classifier.fit(X_train, y_train)
        _, _ = knn_classifier.predict(X_test) #this is like a train step
        for j, (k, predictions) in enumerate(knn_classifier.predict_for_krange(min_k, max_k)):
            
            for y_true, y_pred in zip(y_test, predictions):
                matrices[j][encoding[y_true], encoding[y_pred]] += 1
                if y_true == y_pred:
                    accuracies[i,j] += 1
            
            accuracies[i,j] /= len(y_test)

    avg_accuracies = [sum(accuracies[:,i]) / nb_folds for i in range(accuracies.shape[1])] #array of average accuracies for k in [min_k, max_k]
    return matrices, accuracies, avg_accuracies



            
def precision_recall_by_class(matrix):
    precisions = []
    recalls = []
    assert(matrix.shape[0] == matrix.shape[1])
    for i in range(matrix.shape[0]):
        true_pos = matrix[i,i]
        if (column_sum := sum(matrix[:,i])) != 0:
            precisions.append(true_pos / column_sum)
        else:
            precisions.append(0.)
        if (row_sum := sum(matrix[i,:])) != 0:
            recalls.append(true_pos / row_sum)
        else:
            precisions.append(0.)
    return precisions, recalls


def average_recall_precision(matrix, average="macro"):
    precisions, recalls = precision_recall_by_class(matrix)
    if average == "macro":
        return sum(precisions) / len(precisions), sum(recalls) / len(recalls) if len(recalls) > 0 else 0.
    elif average == "micro": #this will return the same number twice
        true_pos = np.diag(matrix)
        sum_rows = np.sum(matrix, axis=0)
        sum_cols = np.sum(matrix, axis=1)
        false_pos = np.array([x - true_pos[i] for i,x in enumerate(sum_rows)])
        false_neg = np.array([x - true_pos[i] for i,x in enumerate(sum_cols)])
        return sum(true_pos) / (sum(true_pos) + sum(false_pos)), sum(true_pos) / (sum(true_pos) + sum(false_neg))
    else:
        print(f"Invalid arg: {average}")
    
def f_measure(precision, recall):
    return 2*precision*recall / (precision + recall) if (precision + recall) != 0 else 0.

def pf(matrix, average="macro"): 
    true_pos = np.diag(matrix)
    true_neg = np.array([sum([true_pos[i] for i in range(len(true_pos)) if i != j]) for j in range(len(true_pos))])
    sum_rows = np.sum(matrix, axis=0)
    false_pos = np.array([x - true_pos[i] for i,x in enumerate(sum_rows)])
    pf_scores = np.array([x / (x+y) for x,y in zip(false_pos, true_neg)])
    if average == "macro":
        return pf_scores, sum(pf_scores) / len(pf_scores) if len(pf_scores) > 0 else 0.

# def confusion_matrix_str(matrix, X):


def calculate_confusion_matrix(y_true, y_pred):
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
    
def plot_confusion_matrix(confusion_matrix, class_names, precisions, recalls, f_measures, title = None):
    fig, ax = plt.subplots(2,1, figsize = (11, 6), dpi=80)
    cax = ax[0].matshow(confusion_matrix, cmap='Blues')
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax[0].text(j, i, str(confusion_matrix[i, j]), va='center', ha='center')
    
    # Annotate errors of commission and omission
    # for i in range(len(class_names)):
    #     error_of_commission = sum(confusion_matrix[i, :]) - confusion_matrix[i, i]
    #     error_of_omission = sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
        
    #     ax.text(len(class_names), i, f'{error_of_omission}', va='center', ha='center', color='red')
    #     ax.text(i, len(class_names), f'{error_of_commission}', va='center', ha='center', color='red')

    ax[0].set_xticks(np.arange(len(class_names)))
    ax[0].set_yticks(np.arange(len(class_names)))
    ax[0].set_xticklabels(class_names)
    ax[0].set_yticklabels(class_names)
    ax[0].set_xlabel('Predicted label', color = 'blue')
    ax[0].set_ylabel('True label', color = 'blue')
    ax[0].xaxis.set_ticks_position('top')
    ax[0].xaxis.set_label_position('top')
    ax[0].set_title("Confusion Matrix" if not title else title)
    ax[0].set_aspect("auto")


    precisions = [f"{num:.3f}" for num in precisions]
    recalls = [f"{num:.3f}" for num in recalls]
    f_measures = [f"{num:.3f}" for num in f_measures]
    data = [precisions, recalls, f_measures]

    ax[1].axis('off')
    table = ax[1].table(cellText=data, rowLabels=['Precision', 'Recall', 'F-Measure'], colLabels=class_names, cellLoc = 'center', loc='center')
    #table.auto_set_font_size(False)
    #table.set_fontsize(14)
    table.scale(1.2, 1.2)


    plt.tight_layout(rect=[0, 0.00, 1, 0.95])
    
    #plt.figure(figsize=(10, 6), dpi=80)
    #plt.tight_layout()
    #plt.show()


    





