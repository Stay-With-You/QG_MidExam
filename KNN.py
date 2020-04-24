import csv
import random

# 数据读取
with open("E:\\All_Test_Files\\Py_File\\data\\Iris.csv", 'r') as Iris:
	reader = csv.DictReader(Iris)
	data = [row for row in reader]

# 分组
random.shuffle(data)  # 打乱顺序
n = len(data) // 3
test_set = data[0:n]
train_set = data[n:]

# knn
# 欧式距离
def distance(d1, d2):
	res = 0
	for key in ("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"):
		res += (float(d1[key]) - float(d2[key]))**2

	return res**(0.5)

k = 3
def KNN(data):
	# 距离
	res = [
			{"result" : train['Species'], "distance" : distance(data, train)}
			for train in train_set
		  ]
	# print(res)  # check

	# 升序排列
	res = sorted(res, key = lambda item : item['distance'])

    # 取前 k 个
	res02 = res[0:k]
	# print(res02)

    # 加权平均
	result = {'Iris-setosa':0, 'Iris-versicolor':0, 'Iris-virginica':0}

	# 总距离
	sum = 0
	for dis in res02:
		sum += dis['distance']

	for dis02 in res02:
		result[dis02['result']] += 1 - dis['distance']/sum

	# print(result)
	# print("This species: ", data['Species'])

	if (result['Iris-setosa'] > result['Iris-versicolor']) & (result['Iris-setosa'] > result['Iris-virginica']):
		return 'Iris-setosa'
	elif (result['Iris-virginica'] > result['Iris-versicolor']) & (result['Iris-virginica'] > result['Iris-setosa']):
		return 'Iris-virginica'
	elif (result['Iris-versicolor'] > result['Iris-virginica']) & (result['Iris-versicolor'] > result['Iris-setosa']):
		return 'Iris-versicolor'
	else: return -1

# 测试
correct = 0
for test in test_set:
	result = test['Species'] # 真实结果
	result02 = KNN(test)

	if result == result02:
		correct += 1

print(correct)
print(len(test_set))
print("成功率：", (correct / len(test_set)) * 100, "%")

#######################################





