
from __future__ import division, print_function, unicode_literals

import os
import sys
import tarfile
from six.moves import urllib
import hashlib

import numpy as np
import pandas as pd


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
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
##设置随机种子数  每次保证初始化一样
np.random.seed(42)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_SAVE_PATH):
    print("fetch housing data")
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#返回一个包含所有数据的pandas数据结构
def load_housing_data(housing_path=HOUSING_SAVE_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    shuffled_inices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_inices[:test_set_size]
    train_inices = shuffled_inices[test_set_size:]
    return data.iloc[train_inices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.init64(identifier)).digest())[-1] < 256

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

if __name__ == '__main__':
##获取数据集
    ##fetch_housing_data()


# 显示所有列
    pd.set_option('display.max_columns', None)
# 显示所有行
    pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)

    housing = load_housing_data()

    # print(housing.head(10))
    #
    # print(housing.info())
    #
    # print(housing["ocean_proximity"].value_counts())
    #
    # print(housing.describe())
    #
    # housing.hist(bins=50, figsize=(20, 15))
    # plt.show()

    print(np.random.permutation(10))
    train_set, test_set = split_train_test(housing, 0.2)
    print(len(train_set), "train + ", len(test_set), "test")