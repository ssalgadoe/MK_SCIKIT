import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm, neighbors, linear_model, naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris, load_wine, load_boston



features, labels = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

model_nb = make_pipeline(PCA(n_components=4), GaussianNB())
model_nb.fit(X_train, y_train)
prediction =  model_nb.predict(X_test)

model_nb_scale = make_pipeline(StandardScaler(), PCA(n_components=4), GaussianNB())
model_nb_scale.fit(X_train, y_train)
prediction_scaled =  model_nb_scale.predict(X_test)

print(len(features[0]))

accuracy = accuracy_score(prediction, y_test)
print('accuracy of model_nb:', accuracy)


accuracy = accuracy_score(prediction_scaled, y_test)
print('accuracy of model_nb_scaled:', accuracy)

pca = model_nb_scale.named_steps['pca']
scaler = model_nb_scale.named_steps['standardscaler']


#x_transformed = model_nb_scale.transform(scaler.transform(train_x))
X_train_std = pca.transform(scaler.transform(X_train))

FIG_SIZE = (10,10)
#fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)
fig = plt.figure("test", figsize=FIG_SIZE)
#(ax1,ax2) = Axes3D(fig,  elev=45, azim=130)
#ax = fig.(1,1, projection='3d')
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    x = X_train[y_train == l, 0]
    y = X_train[y_train == l, 1]
    z = X_train[y_train == l, 2]
    ax1.scatter(x, y, z, color=c, marker=m, alpha=0.5)

for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    x = X_train_std[y_train == l, 0]
    y = X_train_std[y_train == l, 1]
    z = X_train_std[y_train == l, 2]
    ax2.scatter(x, y, z, color=c, marker=m, alpha=0.5)

ax1.set_title("Training set after PCA")
ax2.set_title("Training set after PCA with standardizer")

for a in (ax1, ax2):
    a.set_xlabel('pca component 1')
    a.set_ylabel('pca component 2')
    a.set_zlabel('pca component 3')
    a.grid()

plt.show()
