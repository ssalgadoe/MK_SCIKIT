import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris


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
clf.fit(train_x, train_y)

correct = 0
for i in range(len(test_y)):
    result = clf.predict(test_x[i].reshape(1,-1))
    if result == test_y[i]:
        correct+=1


print('accuracy', correct/len(test_y))




