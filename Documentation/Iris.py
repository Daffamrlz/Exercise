import pandas as pd
from sklearn.datasets import load_iris
 
# membaca file iris.csv
iris = pd.read_csv('Iris.csv')
# melihat informasi dataset pada 5 baris pertama
iris.head()
# Melihat informasi dataset
iris.info()
# menghilangkan kolom yang tidak penting
iris.drop('Id',axis=1,inplace=True)
# memisahkan atribut dan label
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]]
y = iris['Species']
# Membagi dataset menjadi data latih & data uji
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)