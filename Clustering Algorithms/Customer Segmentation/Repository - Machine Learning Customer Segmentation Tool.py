import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

################################################################################################################################
############################################################ K-MEANS ###########################################################
df = pd.DataFrame(pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv'))
df.drop('Address',axis=1,inplace=True)

x = df.values[:,1:]
x = np.nan_to_num(x)
StandardScaler().fit_transform(x)
k_means = KMeans(init='k-means++',n_clusters=3,n_init=12)
k_means.fit(x)
k_labels = k_means.labels_
df['Cluster'] = k_labels

area = np.pi*(x[:,1]**2)
plt.scatter(x[:,0],x[:,3],s=area,c=k_labels.astype(np.float),alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()
plt.close()

fig = plt.figure(1,figsize=(8,6))
ax = Axes3D(fig,rect=(.0,.0,.95,1.0),elev=50,azim=134)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(x[:,1],x[:,0],x[:,3],c=k_labels.astype(np.float))
plt.savefig('pic.png',bbox_inches='tight')
plt.legend()
plt.show()
plt.close()
