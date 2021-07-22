#导入绘图库
import matplotlib.pyplot as plt
#导入k-means聚类算法
from sklearn.cluster import KMeans
#导入聚类数据生成工具
from sklearn.datasets import make_blobs

#用sklearn自带的make-blobs方法生成聚类测试数据
n_samples = 3000
#该聚类数据集一共1500个样本
X,y = make_blobs(n_samples=n_samples)
print("X:",X)
print("y:",y)
#进行聚类，3类
y_pred = KMeans(n_clusters=3).fit_predict(X)
#用点状图显示聚类效果
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()












