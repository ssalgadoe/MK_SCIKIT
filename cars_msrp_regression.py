import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, neighbors, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

df = pd.read_csv('../data_sets/car_price.csv')

df = pd.get_dummies(df)
df = df.dropna(axis=0, how='any')
df_x = df.drop(['MSRP'], axis=1)
df_y = df['MSRP']

#scaled_x = StandardScaler().fit_transform(df_x.values)
X = np.array(df_x)
y = np.array(df_y)
train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.2)

model = make_pipeline(StandardScaler(),PCA(n_components=5),linear_model.LinearRegression())
model.fit(train_x, train_y)
prediction = model.predict(test_x)
print("r2 score %.2f"  % r2_score(prediction, test_y))
for i in range(100,120):
    print("y: {0} and result: {1}".format(test_y[-i], prediction[-i]))