from Classifier import DecisionTreeClassifier
import pandas as pd
import sys

def main():
    argc = len(sys.argv)
    if argc != 3:
        print("python classify.py <CSVFile> <JSONFile>")
        sys.exit()
    
    csv_file = sys.argv[1]
    json_file = sys.argv[2]

    try:
        data = pd.read_csv(csv_file)
    except:
        print(f"Error: Could not read dataset: {csv_file}")

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

    c = DecisionTreeClassifier("categorical")
    c.from_json(json_file)

    predictions = c.predict(X)
    #print(predictions)

    

    correct = 0
    for y_true, y_pred in zip(y,predictions[0]):
        if y_true == y_pred:
            correct += 1
    print("Accuracy:", correct/len(y))

if __name__=="__main__":
    main()