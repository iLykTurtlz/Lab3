from Classifier import DecisionTreeClassifier
import sys
import pandas as pd
from preprocessing import get_data



def main():
    argc = len(sys.argv)
        
    if argc < 2:
        print("python InduceC45.py <TrainingSetFile.csv> [<restrictionsFile>]")
        sys.exit()
        
    training = sys.argv[1]
    restrictions = sys.argv[2] if argc > 2 else None
    
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
                print(f"Error: {sys.argv[2]} does not have the correct number of columns.")
                sys.exit()
        except:
            print("Restrictions file unreadable")
            sys.exit()
    
    X=None
    y=None
    try:
        # class_col = data.iloc[1,0]
        # data = data.drop(index=[0,1])
        # X = data.loc[:,data.columns != class_col]
        # y = data[class_col]
        X, y = get_data(data)
    except pd.errors.ParserError:
        print("Could not read metadata")
        sys.exit()
    except Exception as e:
        print(e)
        sys.exit()

    if restrictions:
        drop_cols = [col for col, drop in zip(X.columns, restrictions) if not drop]
        X = X.drop(columns=drop_cols)

    tree = DecisionTreeClassifier("complete")
    tree.fit(X, y, threshold=0.01, ratio=True)
    tree_repr = tree.to_json(training, "tree.json", True)
    print(tree_repr)


if __name__ == "__main__":
    main()



