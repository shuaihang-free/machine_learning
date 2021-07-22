#导入logistic回归算法
from sklearn.linear_model import LogisticRegression
#sklearn库带有之ingde鸟尾花数据集，是一个分类问题的数据集
from sklearn.datasets import load_iris
#载入鸟尾花数据集
X,y = load_iris(return_X_y = True)
#训练模型
clf = LogisticRegression().fit(X,y)
#使用模型进行分类预测
pre = clf.predict(X)
print("pre:",pre)
#模型自带的性能评估器，得分
score = clf.score(X,y)
print("score:",score)
