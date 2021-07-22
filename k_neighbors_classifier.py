from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

X,y = load_iris(return_X_y = True)
print("X:",X)
print("y:",y)
clf = KNeighborsClassifier().fit(X,y)
clf.predict(X)

#使用模型进行分类预测
x_ = [[6.5 ,4.1, 5.2, 3.0]]
pre = clf.predict(x_)
print("pre:",pre)
#模型自带的性能评估器，得分
score = clf.score(X,y)
print("score:",score)



