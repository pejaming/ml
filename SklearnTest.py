# -*-coding:utf-8-*-

from sklearn import datasets

iris = datasets.load_iris() # 导入数据集
X = iris.data # 获得其特征向量
y = iris.target # 获得样本label

# from sklearn.datasets.samples_generator import make_classification
#
# X, y = make_classification(n_samples=6, n_features=5, n_informative=2,
#     n_redundant=2, n_classes=2, n_clusters_per_class=2, scale=1.0,
#     random_state=20)
#
# # n_samples：指定样本数
# # n_features：指定特征数
# # n_classes：指定几分类
# # random_state：随机种子，使得随机状可重
#
# for x_,y_ in zip(X,y):
#     print(y_,end=': ')
#     print(x_)

# #数据预处理
# from sklearn import preprocessing
#
# data = [[0, 0], [0, 0], [1, 1], [1, 1]]
# # 1. 基于mean和std的标准化
# scaler = preprocessing.StandardScaler().fit(train_data)
# scaler.transform(train_data)
# scaler.transform(test_data)
#
# # 2. 将每个特征值归一化到一个固定范围
# scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_data)
# scaler.transform(train_data)
# scaler.transform(test_data)
# #feature_range: 定义归一化范围，注用（）括起来


# 作用：将数据集划分为 训练集和测试集
# 格式：train_test_split(*arrays, **options)
from sklearn import model_selection



# """
# 参数
# ---
# arrays：样本数组，包含特征向量和标签
#
# test_size：
# 　　float-获得多大比重的测试样本 （默认：0.25）
# 　　int - 获得多少个测试样本
#
# train_size: 同test_size
#
# random_state:
# 　　int - 随机种子（种子固定，实验可复现）
# 　　
# shuffle - 是否在分割之前对数据进行洗牌（默认True）
#
# 返回
# ---
# 分割后的列表，长度=2*len(arrays),
# 　　(train-test split)
# """
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.linear_model import LogisticRegression
# 定义逻辑回归模型

model = LogisticRegression(penalty="l2", dual=False, tol=0.0001, C=1.0,
    fit_intercept=True, intercept_scaling=1, class_weight=None,
    random_state=None, solver="liblinear", max_iter=100, multi_class="ovr",
    verbose=0, warm_start=False, n_jobs=1)
#
# """参数
# ---
#     penalty：使用指定正则化项（默认：l2）
#     dual: n_samples > n_features取False（默认）
#     C：正则化强度的反，值越小正则化强度越大
#     n_jobs: 指定线程数
#     random_state：随机数生成器
#     fit_intercept: 是否需要常量
# """


# 拟合模型
model.fit(X_train, y_train)
# 模型预测
print(model.predict(X_test))

# 获得这个模型的参数1 
model.get_params()
# 为模型进行打分
#model.score(data_X, data_y) # 线性回归：R square； 分类问题： acc


#保存模型方法1
import pickle

# 保存模型
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)

# 读取模型
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)
model.predict(X_test)


#保存模型方法2
from sklearn.externals import joblib

# 保存模型
joblib.dump(model, 'model.pickle')

#载入模型
model = joblib.load('model.pickle')