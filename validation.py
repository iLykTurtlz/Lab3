import numpy as np
import pandas as pd

def k_folds(X : pd.core.frame.DataFrame, y):
    shuffled_idx = np.random.permutation(X.index)
    shuffled_X = X.reindex(shuffled_idx)
    shuffled_y = y.reindex(shuffled_idx)
    
