# 聚类算法与分类算法的区别是前者是无监督学习（无标签），后者是监督学习（有标签）
# K-means算法即为聚类算法的一种，K为几即聚几类（初始中心（质心)点的个数)
#   ；means即不同的平均数,即中心点到该类其他数据点的平均距离
# 在大规模的数据集上，该算法的收敛速度是比较慢的！

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn')
import seaborn as sns
sns.set_style("whitegrid")

'''
iris = pd.read_csv("E:\\All_Test_Files\\Py_File\\data\\Iris.csv", header = 0)
print(iris.head(6))
print("\n")



# 原始数据散点图
plt.scatter(iris.iloc[:, 1], iris.iloc[:, 2], c = "red", label='See')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title("Species")
plt.legend(loc=2)
plt.show( )


plt.scatter(iris.iloc[:, 3], iris.iloc[:, 4], c = "blue", label='See')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(loc=2)
plt.title("Species")
plt.show( )



# 通过 sklearn 的 k-means 进行训练过的学习效果
from sklearn.cluster import KMeans

X = iris.iloc[:, 0:5]

estimator = KMeans(n_clusters=3)#构造聚类器
estimator.fit(X)#聚类
label_pred = estimator.labels_ #获取聚类标签

#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

plt.scatter(x0.iloc[:, 1], x0.iloc[:, 2], c = "red", marker='o', label='Setosa')
plt.scatter(x1.iloc[:, 1], x1.iloc[:, 2], c = "green", marker='*', label='Virginica')
plt.scatter(x2.iloc[:, 1], x2.iloc[:, 2], c = "blue", marker='+', label='Versicolor')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title("Species")
plt.legend(loc=2)
plt.show()

plt.scatter(x0.iloc[:, 3], x0.iloc[:, 4], c = "red", marker='o', label='Setosa')
plt.scatter(x1.iloc[:, 3], x1.iloc[:, 4], c = "green", marker='*', label='Virginica')
plt.scatter(x2.iloc[:, 3], x2.iloc[:, 4], c = "blue", marker='+', label='Versicolor')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title("Species")
plt.legend(loc=2)
plt.show()
'''

# 手动实现
iris = pd.read_csv("E:\\All_Test_Files\\Py_File\\data\\iris.txt", header = None)
# print(iris.head())
# print(iris.shape)

# 计算欧式距离
def distance(arrA, arrB):
	d = arrA - arrB
	dist = np.sum(np.power(d, 2), axis = 1)   # 对列求和
	return dist

# 随机生成 k 个质心
def randCent(dataset, k):
	n = dataset.shape[1]
	data_min = dataset.iloc[:, :n-1].min()
	data_max = dataset.iloc[:, :n-1].max()
	# 在列中的最小值和最大值之间选取随机质心
	data_cent = np.random.uniform(data_min, data_max, (k, n - 1))
	# 几个特征就几列
	return data_cent
# iris_cent = randCent(iris, 3)
# print(iris_cent)

def KMeans(dataset, k, distMeans = distance, createCent = randCent):
	m, n = dataset.shape
	centroids = createCent(dataset, k)
	clusterAss = np.zeros((m, 3))  # 初始化
	clusterAss[:, 0] = np.inf # 第 0 列的距离初始化为无穷大
	clusterAss[:, 1: 3] = -1
	# 拼接数据
	result_set = pd.concat([dataset, pd.DataFrame(clusterAss)], axis = 1, ignore_index = True)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			# 计算距离
			dist = distMeans(dataset.iloc[i, :n - 1].values, centroids)
			result_set.iloc[i, n] = dist.min()
			result_set.iloc[i, n + 1] = np.where(dist == dist.min())[0]
		# 取反：
		clusterChanged = not (result_set.iloc[:, -1] == result_set.iloc[:, -2]).all()
		# 最后一列放的是上一次迭代结果，倒数第二列放的是本次迭代结果，比较他们的误差是否足够小（相等）：
		if clusterChanged:
			cent_df = result_set.groupby(n + 1).mean() # 质心更新
			centroids = cent_df.iloc[:, :n-1].values
			result_set.iloc[: , -1] = result_set.iloc[:, -2]  # 迭代结果更新
	return centroids, result_set

iris_cent, iris_result = KMeans(iris, 3)
#print(iris_cent)
#print(iris_result.iloc[:, -1].value_counts())

'''
# 模型评估
print("误差平方和：", iris_result.iloc[:, 5].sum())

def KMLearningCurve(dataset, cluster = KMeans, k = 20):
	n = dataset.shape[1]
	SSE = []
	for i in range(1, k):
		centroids, result_set = cluster(dataset, i + 1)
		SSE.append(result_set.iloc[:, n].sum())
	plt.plot(range(2, k + 1), SSE, '--o')
	plt.show()
	return SSE

KMLearningCurve(iris)
'''