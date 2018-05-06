# -*-coding:utf-8-*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression



path = r'wuxian.xlsx'   #此处为脱敏处理，实则是数据文件路径

#df_w是文科生原始数据
#df_l是理科生原始数据
df = pd.read_excel(path,sheet_name=0)
#df_l = pd.read_excel(path,sheet_name=1)
#print(df)

#g_xueke = df.groupby('性别')
#g_xueke['姓名'].count().sort_values(ascending=False)

#print(df)

X = df.as_matrix()[:, 1:] #把DataFrame转换成ndarray，并且进行切片操作，X是数据特征，包括语数英成绩、学校、总分等
y = df.as_matrix()[:, 0]   #同样操作，y是标签，也就是“综合”分数

X_train, X_test, y_train, y_test = train_test_split(X,y) #切分成训练集和测试集

model = LinearRegression() #建立模型
model.fit(X_train, y_train) #训练模型

#print(df.corr())
y_pred = model.predict(X_test)
print(y_pred)

y1 = y_pred.reshape(-1,1)
y2 = y_test.reshape(-1,1)
var1 = pd.DataFrame(y1)
var2 = pd.DataFrame(y2)
var3 = var1 - var2
var = pd.concat([var1,var2,var3],axis=1)
var.columns =['预测分','实际分','误差']
print(var.head())
