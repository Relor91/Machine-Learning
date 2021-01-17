import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
################################################################################################################################
######################################################### SUPPORT VECTOR MACHINE ###############################################
df = pd.DataFrame(pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cell_samples.csv'))
########LETS PLOT THE FIRST 50 ROWS OF DATA FOR CLUMP & UNITSIZE
print(df['Class'].value_counts()) #We can see 2 & 4 refer to Benign & Malignant
ax0 = df[df['Class']==4][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='DarkBlue',label='Malignant')
df[df['Class']==2][0:50].plot(kind='scatter',x='Clump',y='UnifSize',color='yellow',label='Benign',ax= ax0)
plt.show()
print(df.dtypes) #We can see that BareNuc is an object
#########LETS TRANSFORM 'BareNuc' COLUMN TO NUMERIC VALUES AND DISCARD NAN
df = df[pd.to_numeric(df['BareNuc'],errors='coerce').notnull()] #coerce returns NAN if any error, and NOTNULL creates a list of Booleans True if not null value
#########FEATURE MATRIX & SUPPORT VECTOR
x = np.asarray(df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
y = np.asarray(df['Class'])
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
#########LETS USE STANDARD VECTOR MACHINE
svm = svm.SVC(kernel='poly')
svm.fit(x_train,y_train)
yhat = svm.predict(x_test)
#########EVALUATE - CONFUSION MATRIX
try:
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
except:
    pass

#########CONFUSION MATRIX
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)
print (classification_report(y_test, yhat))
#########PLOT NON NORMALIZED CONFUSION MATRIX
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
plt.show()
#########EVALUATE - F1 SCORE & JACCARD SCORE
print('F1 SCORE: ',f1_score(y_test,yhat,average='weighted'))
print('Jaccard Score: ',jaccard_score(y_test,yhat,average='weighted'))
