
from __future__ import division, print_function, unicode_literals

import os
import sys
import tarfile
from six.moves import urllib
import hashlib

import numpy as np
import pandas as pd

from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib


import matplotlib.image as mpimg


from scipy.stats import randint
from scipy import stats
from scipy.stats import geom, expon

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

SAVE_DIR = os.path.dirname(os.path.realpath(__file__))
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_SAVE_PATH = SAVE_DIR
HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"


# 设定图片保存路径，这里写了一个函数，后面直接调用即可
PROJECT_ROOT_DIR = SAVE_DIR
CHAPTER_ID = "02_End_to_End_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "..\\images", CHAPTER_ID)
##设置随机种子数  每次保证初始化一样
np.random.seed(42)

##  列索引
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
##  定制转换器
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        ##  每个屋子的平均房间数
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        ##  每个家庭的平均人数
        population_per_household = X[:, population_ix] / X[:, household_ix]
        ##  计算每个房间的平均床数
        if self.add_bedrooms_per_room:
            bedrooms_per_rooms = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_rooms]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

##  原始数据转换选择器   选择 数值列还是 文字类别列
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values





def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


##  获取数据
def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_SAVE_PATH):
    print("fetch housing data")
    ##  目录不存在 生成新目录
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    ## 设置下载路径
    tgz_path = os.path.join(housing_path, "housing.tgz")
    ## 下载文件
    urllib.request.urlretrieve(housing_url, tgz_path)
    ##  打开压缩文件文件
    housing_tgz = tarfile.open(tgz_path)
    ##  解压文件
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#返回一个包含所有数据的pandas数据结构
def load_housing_data(housing_path=HOUSING_SAVE_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#拆分训练和测试数据集
def split_train_test(data, test_ratio):
    ##  将数组索引 打散
    shuffled_inices = np.random.permutation(len(data))
    ## 计算测试数据大小
    test_set_size = int(len(data) * test_ratio)
    ## 前一半数据为 测试数据
    test_indices = shuffled_inices[:test_set_size]
    ##  剩下的为训练数据
    train_inices = shuffled_inices[test_set_size:]
    ##  按照索引返回训练数据和测试数据
    return data.iloc[train_inices], data.iloc[test_indices]

##根据id的MD5的最后一位字符  按比例计算测试集
def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    ## 该id是否为测试集
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256*test_ratio
## 利用id 拆分训练数据和测试数据
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]   ##  id_column  为 "index"
    ##  将id list 转换为 Boolean list
    in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# 收入类别比例
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())




if __name__ == '__main__':
    ##获取数据集
    ##fetch_housing_data()


# # 显示所有列
    pd.set_option('display.max_columns', None)
 # 显示所有行
    pd.set_option('display.max_rows', None)
 # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)
     ## 读取住房数据
    housing = load_housing_data()
     ##  查看数据前 n 行数据
    print("line = 167 head 10:\n", housing.head(10), "\n")
     #  查看数据详情
    print("line = 169 info:\n", housing.info(), "\n")
     # 数据集 housing  中 每一项值的统计
    print("line = 171 housing[\"ocean_proximity\"].value_counts()\n", housing["ocean_proximity"].value_counts(),"\n")
    #  数据属性摘要
    print("line = 173 housing.describe()\n", housing.describe(), "\n")
    # 直方图列表  bins 指定网格的密度
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()

#
    # 利用伪随机 拆分训练集和测试机
    train_set, test_set = split_train_test(housing, 0.2)
    print("line = 187 print train size and  test size \t", len(train_set), "train + ", len(test_set), "test\n")
    print("line = 188 print  test data  head n = \n", test_set.head(), "\n")
#
    ##  利用hash 拆分训练集和测试机
    ##  还原索引
    housing_with_id = housing.reset_index()
    ##使用行索引作为ID
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

    ##  自定义索引 拆分测试集和训练集
    housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    print("line = 202 self define for test:\n", test_set.head(), "\n")

    ## 利用scikit-learn 提供的函数
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    print("line = 206  split by scikit-leaarn func :\n", test_set.head(), "\n")

    # 收入中位数直方图 原始图
    housing["median_income"].hist()
    plt.show()

    # 除以1.5 以限制收入类别的数量
    housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    ## 对收入中位数按照 1万 2万 3万 4万 5万及5万以上进行统计
    print("line = 216 income cat value_counts(): \n",housing["income_cat"].value_counts(),"\n")
    # 调整后的直方图
    housing["income_cat"].hist()
    plt.show()

    ##  根据收入类别进行分层抽样  循环一次
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

##    完整数据集中 查看收入类别比例
    print("line = 228 all cat \n", housing["income_cat"].value_counts() /len(housing), "\n")
    # 测试集中 查看收入类别比例
    print("line = 230 test cat \n", strat_test_set["income_cat"].value_counts() / len(strat_test_set), "\n")


    ##  拆分测试集和训练集
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }).sort_index()

    #随机误差
    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    # 分层抽样的误差
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

    print("line = 247 compare props\n", compare_props, "\n")

    # 删除income_cat 属性 恢复原始状态
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # 创建一个 训练数据副本
    housing = strat_train_set.copy()

    ##可视化 散点 信息
    housing.plot(kind="scatter", x="longitude", y="latitude")
    plt.show()

    # 设置显示密度
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()

    ## 房价和人口关系图
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"),
                 colorbar=True,
                 sharex=False)
    plt.legend()
    save_fig("housing_prices_scatterplot")
    plt.show()


    #  读取前一张图片 将其放到加利福利亚的地图上
    california_img = mpimg.imread(PROJECT_ROOT_DIR + '\\..\\images\\02_End_to_End_project\\california.png')
    ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10, 7),
                      s=housing['population']/100, label="Population",
                      c="median_house_value", cmap=plt.get_cmap("jet"),
                      colorbar=False,
                      alpha=0.4
                      )
    ##  底图显示加州地图
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.08], alpha=0.5,
               cmap=plt.get_cmap("jet"))
    ##  设置 横轴 纵轴
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)
    ##  读取房屋价中位数
    prices = housing["median_house_value"]
    ##  生成由低至高的 价格矩阵
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    ##  设置色盘
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
    cbar.set_label('Median House Value', fontsize=16)
    ##  设置图例
    plt.legend(fontsize=16)
    save_fig("california_housing_prices_plot")
    plt.show()


    # 使用 corr()  计算每对属性之间的标准相关性系数
    corr_matrix = housing.corr()
    # 查看每个属性与房屋中值的相关程度
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    ## 绘制部分属性 相关性图表
    attributes = ["median_house_value",
                  "median_income",
                  "total_rooms",
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    save_fig("scatter_matrix_plot")
    plt.show()

    ## 房屋中值与 收入中位数 的相关性散点图
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.axis([0, 16, 0, 550000])
    save_fig("income_vs_house_value_scatterplot")
    plt.show()

    ## 每个家庭平均拥有的房间数 = 总房间数 / 总的家庭数
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    ## 总卧室数 / 总房间数
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    ##  平均家庭人口数
    housing["population_per_household"] = housing["population"]/housing["households"]

    ## 查看新的属性的相关性
    corr_matrix = housing.corr()
    print("line = 332 corr_matrix[\"median_house_value\"]\n",
          corr_matrix["median_house_value"].sort_values(ascending=False),"\n")
    ## 绘制散点图
    housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)
    plt.axis([0, 5, 0, 520000])
    plt.show()

    print("line = 339  housing.describe\n", housing.describe(), "\n")

## 准备数据
    ## 将 median_house_value_ 这一列的属性去掉 同时创建数据副本 作为训练集
    housing = strat_train_set.drop("median_house_value", axis=1)
    ## 创建数据集标签 median_house_value 即为标签
    housing_labels = strat_train_set["median_house_value"].copy()

    ## 抽样取出部分新创建的训练集中的实例
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    print("line = 349 sample_incomplete_rows \n",sample_incomplete_rows, "\n")


##  data cleaning 数据清洗
    ## option 1-删除相应的区域
    sample_incomplete_rows.dropna(subset=["total_bedrooms"])
    ## option 2-删除整个属性
    sample_incomplete_rows.drop("total_bedrooms", axis=1)
    ## 计算中值
    median = housing["total_bedrooms"].median()
    ## 将缺失值 设置为某个值 (零  均值  中位数) 0, data.mean(), median
    sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)
    print("line = 361  sample_incomplete_rows \n", sample_incomplete_rows, "\n")

    ##  使用imputer 处理缺失值
    # 创建了一个实例  并指定使用中位数替换缺失值
    #  median 为 total_bedrooms 的中值 在上一个cell 中已经计算出来 这里直接用
    imputer = Imputer(strategy="median")

    # 移除 ocean_proximity 的数据副本 即把这一列去掉
    housing_num = housing.drop('ocean_proximity', axis=1)

    # 使用 fit 将imputer 实例 fit 到训练数据
    imputer.fit(housing_num)
    ## 打印 imputer 中计算的中位数
    print("line = 374 imputer.statistics_ \n", imputer.statistics_, "\n")

    ## 打印 原始数据的 中位数
    print("line = 377 housing_num.media().values \n", housing_num.median().values, "\n")


    ## 使用 训练好的 imputer 通过学习的中位数 替换缺失值对 训练集进行转换
    X = imputer.transform(housing_num)
    ## 将转换后的numpy 数组 放回 pandas dataFrame
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))
    print("line = 384 housing_tr \n",housing_tr.loc[sample_incomplete_rows.index.values], "\n")
    print("line = 385 imputer.strategy \n", imputer.strategy, "\n")
    ##   全量转换数据
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    print("line = 388 housing.head \n", housing.head(), "\n")

    ## 打印文字属性
    housing_cat = housing[['ocean_proximity']]
    print("line = 392 housing_cat.head() \n", housing_cat.head(10), "\n")

    ##将文字标签转换为 数字
    encoder = LabelEncoder()
    housing_cat = housing["ocean_proximity"]
    housing_cat_encoded = encoder.fit_transform(housing_cat)
    print("line = 398  ocean_proximity housing_cat_encoded\n", housing_cat_encoded, "\n")
    print("line = 399 ocean label \n", encoder.classes_, "\n")

    ## oneHot编码
    encoder = OneHotEncoder()
    housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
    print("line = 404 housing_cat_hot \n", housing_cat_1hot, "\n")
    print("line = 405 husing_cat_1hot.toarray()", housing_cat_1hot.toarray())

    ## 使用labelBinarizer 一次完成这两种转换
    encoder = LabelBinarizer()
    housing_cat_1hot = encoder.fit_transform(housing_cat)
    print("line = 410 housing_cat_1hot \n", housing_cat_1hot, "\n")

    ##  定制转换器  添加 每个屋子 平均房间数 和 平均人数
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)
    ##  将 新增的列 添加到dataFrame中
    housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs,
        columns=list(housing.columns)+["rooms_per_household", "population_per_househlod"])
    print("line = 423 housing_extra_attribs.head() \n",housing_extra_attribs.head(), "\n")



##  transformation pipelines
    num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_addr', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    ##  数值属性的 小型 pipeline
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    print("line = 435 housing_num_tr \n", housing_num_tr, "\n")

    ##  将housing_num 转换为 列表
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    # 将housing 的ocean_proximity 标签 编码
    housing_prepared = full_pipeline.fit_transform(housing)
    print("line = 446 housing_prepared \n", housing_prepared, "\n")
    print("line = 447 housing_prepared.shape = ", housing_prepared.shape, "\n")


    ## 将数值pipeline 和 分类pipeline 转换连接到 单个pipeline 上面

    # 数字属性
    num_attribs = list(housing_num)
    # 分类属性
    cat_attribs = ["ocean_proximity"]

    ##定义数字属性 的 pipeline
    old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    ##  定义分类属性的pipeline
    old_cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])
    ##   数值和文本综合pipeline
    old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])

    old_housing_prepared = old_full_pipeline.fit_transform(housing)
    print("line = 476 old_housing_prepared \n", old_housing_prepared, "\n")
    print("line = 477 housing_prepared and old_housing_prepared is same:",
          np.allclose(housing_prepared, old_housing_prepared), "\n")


## 选择模型并进行训练
    # 训练一个线性回归模型
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    ##  取前5个数字
    some_data = housing.iloc[:5]
    ##  取前5个的标签
    some_labels = housing_labels.iloc[:5]
    ##  将部分数据 进行标准化转换
    some_data_prepared = full_pipeline.transform(some_data)

    print("line = 492 Predictions:\t", lin_reg.predict(some_data_prepared))
    ## 与实际值比较  相差比较大
    print("line = 494 Labels:\t", list(some_labels))
    print("line = 495 some_data_prepared", some_data_prepared)

    ## 使用 mean_squared_error 函数 在整个训练集测量 这个回归模型的RMSE
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("line = 501 lin_rmse =\t", lin_rmse)

    ##  训练 决策树
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print("line = 510 tree_rmse=\t", tree_rmse)


    ## 使用交叉验证进行评估 cv=10  随机分成10个子集 每次在9个子集训练 剩下的1个子集评估
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)

    ## 显示结果
    display_scores(tree_rmse_scores)

    ## 显示用线性回归模型 的得分结果
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                 scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)

    ## 随机森林
    forest_reg = RandomForestRegressor(random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    #  rmse 评估训练结果
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    print("line = 534 forest_rmse = \t", forest_rmse)

    ##  随机深林进行交叉验证
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)

    scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    print("line = 544 forest scores cross_val_score = \t", pd.Series(np.sqrt(-scores)).describe())

    ## 支持向量机
    svm_reg = SVR(kernel="linear")
    svm_reg.fit(housing_prepared, housing_labels)
    housing_predictions = svm_reg.predict(housing_prepared)
    svm_mse = mean_squared_error(housing_labels, housing_predictions)
    svm_rmse = np.sqrt(svm_mse)
    print("line = 552 svm_rmse = \t", svm_rmse)



##  Grid Search 网格搜索
    ##
    param_grid = [
        {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg,
                               param_grid,
                               cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(housing_prepared, housing_labels)
    print("line = 570 grid_search.best_params = \t", grid_search.best_params_)
    print("line = 571 grid_search.best_estimator = \t", grid_search.best_estimator_)

    cvres = grid_search.cv_results_

    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    print("line = 578 grid_search.cv_results_", pd.DataFrame(grid_search.cv_results_))


    ## Randomized Search 随机搜索

    param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                        n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)

    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)



## 分析最佳模型及其错误

    feature_importances = grid_search.best_estimator_
    print("line = 602 feature_importances = \t", feature_importances)

    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    cat_one_attribs = list(cat_encoder.categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_attribs

    print("line = 609 attributes = \t", attributes)
    ###print("line = 610  ", sorted(zip(feature_importances, attributes), reverse=True))
    sorted(zip(feature_importances, attributes), reverse=True)


    ## 在测试集上 测试模型
    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("line = 624 final_rmse = \t ", final_rmse)

    ##  置信区间设为 95%
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    mean = squared_errors.mean()
    m = len(squared_errors)

    print("line = 632 rmse of 95% confidence = \t", np.sqrt(stats.t.interval(confidence, m - 1,
                             loc=np.mean(squared_errors),
                             scale=stats.sem(squared_errors))))
    # 手动计算间隔
    tscore = stats.t.ppf((1 + confidence) /2, df= m -1)
    tmargin = tscore * squared_errors.std(ddof=1)/np.sqrt(m)
    print("line = 638 calculate rmse of 95% confidence = \t", np.sqrt(mean - tmargin), np.sqrt(mean + tmargin))

    # 使用z分数 而不是t分数
    zscore = stats.norm.ppf((1 + confidence) / 2)
    zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
    print("lie = 643 z score = \t", np.sqrt(mean - zmargin), np.sqrt(mean + zmargin))

    ##  完整管道
    full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

    full_pipeline_with_predictor.fit(housing, housing_labels)
    full_pipeline_with_predictor.predict(some_data)

    my_model = full_pipeline_with_predictor
    ##  训练结果 持久化
    joblib.dump(my_model, "my_model.pkl")
    my_model_loaded = joblib.load("my_model.pkl")

    geom_distrib = geom(0.5).rvs(10000, random_state=42)
    expon_distrib = expon(scale=1).rvs(10000, random_state=42)
    plt.hist(geom_distrib, bins=50)
    plt.show()
    plt.hist(expon_distrib, bins=50)
    plt.show()
