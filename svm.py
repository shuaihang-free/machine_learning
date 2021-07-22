from sklearn.datasets import load_iris
from sklearn.svm import SVC

X,y = load_iris(return_X_y = True)
print("X:",X)
print("y:",y)
#训练模型
clf = SVC().fit(X,y)
print("kernel:",clf.kernel)
x_ = [[5.9, 3.0,  5.1, 1.8]]
print("x_:",x_)
#使用模型进行分类预测
pre = clf.predict(x_)
print("pre:",pre)
#模型自带的性能评估器，得分
score = clf.score(X,y)
print("score:",score)



