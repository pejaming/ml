# -*-coding:utf-8-*-


# http://bluewhale.cc/2016-09-02/python-linear-regression-in-scikit-learn.html使用Python训练回归模型并进行预测


# 导入机器学习linear_model库
from sklearn import linear_model
# 导入交叉验证库
from sklearn import cross_validation
# 导入数值计算库
import numpy as np
# 导入科学计算库
import pandas as pd
# 导入图表库
import matplotlib.pyplot as plt

# 读取数据并创建数据表，名称为cost_and_click
cost_and_click = pd.DataFrame(pd.read_excel('cost_and_click.xlsx'))

# 查看数据表前5行的内容
print(cost_and_click.head())

# 将广告成本设为自变量X
X = np.array(cost_and_click[['cost']])
# 将点击量设为因变量Y
Y = np.array(cost_and_click['click'])
# 查看自变量和因变量的行数
X.shape, Y.shape

# 设置图表字体为华文细黑，字号15
plt.rc('font', family='STXihei', size=15)
# 绘制散点图，广告成本X，点击量Y，设置颜色，标记点样式和透明度等参数
plt.scatter(X, Y, 60, color='blue', marker='o', linewidth=3, alpha=0.8)
# 添加x轴标题
plt.xlabel('成本')
# 添加y轴标题
plt.ylabel('点击量')
# 添加图表标题
plt.title('广告成本与点击量分析')
# 设置背景网格线颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.4)
# 显示图表
# plt.show()

# 将原始数据通过随机方式分割为训练集和测试集，其中测试集占比为40%
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)

# 查看训练集数据的行数
X_train.shape, y_train.shape

# 将训练集代入到线性回归模型中
clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)

# 线性回归模型的斜率
clf.coef_

# 线性回归模型的截距
clf.intercept_

# 判定系数R Square
clf.score(X_train, y_train)

# 有了前面的斜率和截距后，就可以描绘出广告成本与广告点击量之间的关系了，并且可以通过这种关系来对广告点击量的结果进行预测，下面我们在回归模型中将广告成本设置为20000元，经过计算获得预测的广告点击量为1993次。这个预测结果准确吗？我们还需要对回归模型进行评估。

# 输入自变量预测因变量
print(clf.predict(21))

# 将测试集的自变量代入到模型预测因变量
list(clf.predict(X_test))
# 显示测试集的因变量
list(y_test)

# 计算误差平方和
((y_test - clf.predict(X_test)) ** 2).sum()
