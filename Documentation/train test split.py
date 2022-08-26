import sklearn
from sklearn import datasets

#load iris dataset
iris = datasets.load_iris()

#mendifinisikan atribut dan label pada dataset
x=iris.data
y=iris.target

print(x)
print(y)