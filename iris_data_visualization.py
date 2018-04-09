import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target

fig, axes = plt.subplots(nrows=3, ncols=2)

axes[0,0].scatter(X[:,0],X[:,1], c=y)
axes[0,0].set_xlabel(data.feature_names[0])
axes[0,0].set_ylabel(data.feature_names[1])

axes[0,1].scatter(X[:,0],X[:,2], c=y)
axes[0,1].set_xlabel(data.feature_names[0])
axes[0,1].set_ylabel(data.feature_names[2])

axes[1,0].scatter(X[:,0],X[:,3], c=y)
axes[1,0].set_xlabel(data.feature_names[0])
axes[1,0].set_ylabel(data.feature_names[3])

axes[1,1].scatter(X[:,1],X[:,2], c=y)
axes[1,1].set_xlabel(data.feature_names[1])
axes[1,1].set_ylabel(data.feature_names[2])

axes[2,0].scatter(X[:,1],X[:,3], c=y)
axes[2,0].set_xlabel(data.feature_names[1])
axes[2,0].set_ylabel(data.feature_names[3])


axes[2,1].scatter(X[:,2],X[:,3], c=y)
axes[2,1].set_xlabel(data.feature_names[2])
axes[2,1].set_ylabel(data.feature_names[3])


plt.show()