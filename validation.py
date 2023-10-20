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
    if k==-1:
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
        assert(sum(len(x) for x in Xs) == len(X) and sum(len(Y) for Y in ys) == len(y))
        fold_indices = [([i],[j for j in range(len(Xs)) if i!=j]) for i in range(len(Xs))]
        return (  ((pd.concat([Xs[j] for j in train]), Xs[test[0]]), (pd.concat([ys[j] for j in train]), ys[test[0]])) for test, train in fold_indices  )



def cross_validation(classifier, X, y, k, threshold, ratio=False):
    encoding = {attribute:number for number,attribute in enumerate(X.columns)}
    n = len(X.columns) + 1
    matrix = np.zeros(shape=(n,n), dtype=int)
    accuracies = np.zeros(k)
    for i, ((X_train, X_test), (y_train, y_test)) in enumerate(k_folds(X,y,k)):
        classifier.fit(X_train, y_train, threshold, ratio)
        predictions = classifier.predict(X_test)
        for y_true, y_pred in zip(y_test, predictions):
            matrix[encoding[y_true], encoding[y_pred]] += 1
            if y_true == y_pred:
                accuracies[i] += 1
        accuracies[i] /= len(y_true)
            
        





