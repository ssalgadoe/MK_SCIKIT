import numpy as np
import pandas as pd
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data_sets/creditcard_sample.csv')

data = np.array(df)

for i in range(len(data)):
    if data[i,-1] == 1:
        d = data[i].reshape(1,-1)
        for k in range(0,40):
            data  = np.append(data,d,axis=0)




print(len(data))

X = data[:,:-1]
y = data[:,-1]

train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.8)


clf = svm.SVC()
clf.fit(train_x,train_y)

result = clf.predict(test_x)
correct =0
negatives =0
f_negative = 0
f_positive = 0
wrong_list = {}
wrong_list[0] = []
wrong_list[1] = []
for i in range(len(result)):
    if result[i]== test_y[i]:
        correct+=1
    elif test_y[i] == 0:
        f_positive+=1
        wrong_list[test_y[i]].append(i)
    elif test_y[i] == 1:
        f_negative+=1
        wrong_list[test_y[i]].append(i)
    if test_y[i]==1:
        negatives +=1

print("accuracy", correct/len(result))
print("wrong_list", len(wrong_list), wrong_list)
print("total negatives {0} false_positives {1} false negatives {2}".format(negatives, f_positive, f_negative))
#
#
# print(test_y[34],result[34])
# print(test_y[199],result[199])

