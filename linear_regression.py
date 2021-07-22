#从sklearn包里导入现行回归模型中的线性回归算法
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
#生成数据集
x = np.linspace(-3,3,30)
x = x + np.random.rand(30) #加扰动
y = 2*x + 1
#数据集绘图
plt.scatter(x,y)
plt.show()

x = [[i] for i in x]
y = [[i] for i in y]
x_ = [[1],[2]]
#训练线性回归模型
model = linear_model.LinearRegression()
model.fit(x,y)

pre = model.predict(x_)
print("pre:",pre)

