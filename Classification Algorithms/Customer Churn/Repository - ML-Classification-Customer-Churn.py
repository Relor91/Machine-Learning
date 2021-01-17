import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
################################################################################################################################
######################################################### LOGISTIC REGRESSION ##################################################
churn_df = pd.DataFrame(pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv'))
churn_df['churn'].astype(int)
#########FEATURE MATRIX AND RESPONSE VECTOR
x = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])
#########NORMALIZATION
sl = StandardScaler()
x = sl.fit_transform(x)
#########SPLIT DATA
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=4)
#########TRAIN AND FIT THE LOGISTIC REGRESSION
lr = LogisticRegression(C=0.01,solver='liblinear')
lr.fit(x_train,y_train)
yhat = lr.predict(x_test)
#########RETURN PROBABILITIES FOR BOTH CLASSES OF RESULTSP (Y=1|X) 1 AND (Y=0|X) 0
yhat_prob = lr.predict_proba(x_test)
#########EVALUATE USING CONFUSION MATRIX
jaccard_score(y_test,yhat,average='weighted')
print('Jaccard Score: ',jaccard_score(y_test,yhat,average='weighted'))
#########EVALUATE WITH CONFUSION MATRIX

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
    print(confusion_matrix(y_test, yhat, labels=[1,0]))
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
    np.set_printoptions(precision=2)



    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
    plt.show()
except:
    pass

#########EVALUATE USING CLASSIFICATION REPORT AND F1 SCORE
print(classification_report(y_test, yhat))
#########EVALUATE USING LOG LOSS
print('Log Loss: ',log_loss(y_test,yhat_prob))