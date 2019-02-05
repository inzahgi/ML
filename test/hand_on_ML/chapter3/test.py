
from __future__ import division, print_function, unicode_literals

import numpy as np
import os



np.random.seed(42)

import  matplotlib
import  matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

##PROJECT_ROOT_DIR = "/home/inzahgi/test/jupyter/hand_on_ML/Hands-on-Machine-Learning"
PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CHAPTER_ID = "03_Classification_MNIST"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "..\\images", CHAPTER_ID)


def save_fig(fig_id, tight_lay=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("line 26 path = ", path)
    print("Saving figure", fig_id)
    if tight_lay:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


import matplotlib
import matplotlib.pyplot as plt

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instances.reshape(size, size) for instances in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original', data_home='./')
    print("line = 37", mnist)

    X, y = mnist["data"], mnist["target"]
    print("line = 40 ", X.shape)
    print("line = 41", y.shape)

    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    ##plt.show()

    print("line=52", y[36000])


    plt.figure(figsize=(9,9))
    example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
    plot_digits(example_images, images_per_row=10)
    ##save_fig("more_digits_plot")
    ##plt.show()

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    ##  划分训练集
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(max_iter=5, random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    ##  用刚才得到的5进行验证
    sgd_clf.predict([some_digit])

    ## 交叉验证集评估模型
    print("line = 102 ",cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

    skfolds = StratifiedKFold(n_splits=3, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train_5[train_index])
        X_test_fold = X_train[test_index]
        y_test_fold = (y_train_5[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))

    ##   对"not-5" 类中的每个图像进行分类  交叉评估
    never_5_clf = Never5Classifier()
    print("line = 131", cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


    ##  confusion matrix 混淆矩阵
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    confusion_matrix(y_train_5, y_train_pred)

    y_train_perfect_predictions = y_train_5
    ## 混淆矩阵
    print("line = 143 ",confusion_matrix(y_train_5, y_train_perfect_predictions))
    ## 精度
    print("line = 145", precision_score(y_train_5, y_train_pred))
    ## 召回率
    print("line = 147", recall_score(y_train_5, y_train_pred))
    ## f1得分
    print("line = 149", f1_score(y_train_5, y_train_pred))
    ##  预测决策分数
    y_scores = sgd_clf.decision_function([some_digit])
    print("line = 152", y_scores)






