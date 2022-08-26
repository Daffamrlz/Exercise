import sklearn
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()

x=iris.data
y=iris.target

#membuat model dengan decision tree classifier
clf = tree.DecisionTreeClassifier()

#mengevaluasi performa model dengan cross_val_score
scores = cross_val_score(clf, x, y, cv = 5)

print(scores)
