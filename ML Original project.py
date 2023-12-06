import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as train_data_split
from sklearn.neighbors import KNeighborsClassifier
#pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_csv("C:\\Users\\Adil\\OneDrive\\Desktop\\ML Project DataBase\\student_placement_data\\collegePlace.csv")
print(df.head())
df.loc[(df['Stream'] == 'Civil') , 'Stream'] = 0
df.loc[(df['Stream'] == 'Computer Science') , 'Stream']  = 1
df.loc[(df['Stream'] == 'Electrical') , 'Stream'] =  2
df.loc[(df['Stream'] == 'Electronics And Communication') , 'Stream'] =  3
df.loc[(df['Stream'] == 'Information Technology') , 'Stream'] =  4
df.loc[(df['Stream'] == 'Mechanical') , 'Stream'] =  5
print(df.head())
#to delete the stream axis and create a new column in df(dataframe)
Sex=df['Gender']
df['Gender'] = Sex.replace(to_replace=['Male', 'Female'],value=[0,1])
#to replace male to 0 and female to 1
df['AgeGroup'] = pd.cut(df['Age'], 5)
df.loc[(df['Age'] > 18) & (df['Age'] <= 21), 'Age'] = 0
df.loc[(df['Age'] > 21) & (df['Age'] <= 23), 'Age'] = 1
df.loc[(df['Age'] > 23) & (df['Age'] <= 25), 'Age'] = 2
df.loc[(df['Age'] > 25) & (df['Age'] <= 27), 'Age'] = 3
df.loc[(df['Age'] > 27), 'Age'] = 4
#dataframe.loc is uesd to  access a group of rows and columns by label(s) or a boolean array.
X = df.drop(['PlacedOrNot', 'AgeGroup'], axis=1)
#to delete AgeGroup and PlacedOrNot
y = df['PlacedOrNot']
#to insert a new dataframe in y(data frame)
X_train, X_test, y_train, y_test = train_data_split(X, y, test_size=0.002)
#sklearn.model_selection.train_test_split(*arrays, test_size=None,random_state=None)
models_accuracy = {}
print(X_test)
#Creating a prediction model on KNN(K NearestNeighbour)
kn_model = KNeighborsClassifier(n_neighbors=10)
kn_model.fit(X_train, y_train)
kn_model.predict(X_test)
print(X_test)
kn_score = kn_model.score(X_test, y_test)
models_accuracy['Knn'] = kn_score*100
print(models_accuracy)
print(kn_model.predict(X_test))
print(y_test.values)