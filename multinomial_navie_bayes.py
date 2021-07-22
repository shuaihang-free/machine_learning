from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB

X,y = load_iris(return_X_y = True)
print("X:",X)
print("y:",y)
clf = MultinomialNB().fit(X,y)
x_ = [[5.9, 3.0,  5.1, 1.8]]
#使用模型进行分类预测
pre = clf.predict(x_)
print("pre:",pre)
#模型自带的性能评估器，得分
score = clf.score(X,y)
print("score:",score)



