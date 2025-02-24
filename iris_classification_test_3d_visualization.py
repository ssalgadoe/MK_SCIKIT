import numpy as np
from sklearn import svm, neighbors
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = load_iris()
X = data.data
Y = data.target
Z = [np.append(x,y) for x, y in zip(X,Y)]
np.random.shuffle(Z)
Z = np.array(Z)

X = Z[:,:-1]
Y = Z[:,-1]

ratio = 0.5
threshold = int(ratio*len(Y))
train_x = X[:threshold]
train_y = Y[:threshold]
test_x = X[threshold:]
test_y = Y[threshold:]

clf = svm.SVC()
clf = neighbors.KNeighborsClassifier()
clf.fit(train_x, train_y)

result = clf.predict(test_x)
correct = 0
for i in range(len(test_y)):
    # result = clf.predict(test_x[i].reshape(1,-1))
    if result[i] == test_y[i]:
        correct+=1

accuracy = [ 5 if result[i]==test_y[i] else 10 for i in range(len(test_y)) ]

print('accuracy', correct/len(test_y))


fig, axes = plt.subplots(nrows=2, ncols=2)

# axes[0,0].scatter(test_x[:,0],test_x[:,1], c=accuracy)
# axes[0,1].scatter(test_x[:,0],test_x[:,2], c=accuracy)
# axes[1,0].scatter(test_x[:,1],test_x[:,2], c=accuracy)
# axes[1,1].scatter(test_x[:,1],test_x[:,3], c=accuracy)


ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

ax.scatter(test_x[:,0],test_x[:,1], test_x[:,2], c=accuracy, s=50)
ax.scatter(test_x[:,0],test_x[:,1], test_x[:,2], c=result, s=10)

plt.show()

