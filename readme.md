#Decision Trees, Random Forests, KNN

Paul Jarski     pjarski@calpoly.edu
Joe Dewar       jdewar@calpoly.edu

Note: incoming csv data files are expected to be in the same format as the originals: with a column giving the domain size for each class and a row dedicated to the label of the class.  Our programs will remove these two rows after extracting their information.


Executable files:

induceC45.py        usage: python3 InduceC45 <TrainingSetFile.csv> [<restrictionsFile>]
                    result: creates and writes a json file, tree.json

classify.py         usage: python3 classify.py <CSVFile> <JSONFile>

evaluate.py         usage: python3 evaluate.py <CSVFile> [<restrictionsFile>] <numberOfFolds or -1 for leave one out>

Output files:

The .out files for each run can be found in the Output folder
We also created a Graphs folder containing confusion matrices for each run with tabulated recall, precision, and f-scores.