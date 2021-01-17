import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pydotplus
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from six import StringIO

################################################################################################################################
######################################################### DECISION TREE ########################################################
df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv')

x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df[['Drug']].values

le = LabelEncoder()
le.fit(['F','M'])
x[:,1] = le.transform(x[:,1])

le = LabelEncoder()
le.fit(['HIGH','LOW','NORMAL'])
x[:,2] = le.transform(x[:,2])

le = LabelEncoder()
le.fit(['NORMAL', 'HIGH'])
x[:,3] = le.transform(x[:,3])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=3)
dt = DecisionTreeClassifier(criterion='entropy',max_depth=4)
dt.fit(x_train,y_train.ravel())
yhat = dt.predict(x_test)
print('Decision Tree accuracy score is ',accuracy_score(y_test,yhat))

dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out = tree.export_graphviz(dt,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
plt.show()

