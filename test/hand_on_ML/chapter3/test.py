
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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


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


## 绘制精度和召回率 作为阈值的函数
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1],
             "b--",
             label="Precision")
    plt.plot(thresholds,
             recalls[:-1],
             "g--",
             label="Recall")
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls,
             precisions,
             "b--",
             linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

## 画出混淆矩阵的图像
def plot_confusion_matrix(matrix):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


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

    ## 阈值判断
    threshold = 200000
    y_some_digit_pred = (y_scores > threshold)
    print("line = 157", y_some_digit_pred)

    ## 获得训练集中所有实例分数
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    print("line = 161", y_scores.shape)

    ## 计算所有可能的精度和召回率的阈值
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    ## 利用matplotlib绘制精度和召回率 作为阈值的图像
    plt.figure(figsize=(8, 4))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.xlim([-700000, 700000])
    ##save_fig("precision_recall_vs_threshold_plot")
    ##plt.show()

    (y_train_pred == (y_scores > 0)).all()

    plt.figure(figsize=(8, 6))
    plot_precision_vs_recall(precisions, recalls)
    ##save_fig("precision_vs_recall_plot")
    ##plt.show()

    y_train_pred_90 = (y_scores > 70000)

    ##  精确度得分
    print("line = 209",precision_score(y_train_5, y_train_pred_90))
    ## 召回率得分
    print("line = 212", recall_score(y_train_5, y_train_pred_90))

    ##  需要使用 roc_curve 函数计算各种阈值的 TPR 和 FPR
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr)
    save_fig("roc_curve_plot")
    plt.show()

    roc_auc_score(y_train_5, y_scores)

    forest_clf = RandomForestClassifier(random_state=42)
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                        method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")

    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.legend(loc="lower right", fontsize=16)
    ##save_fig("roc_curve_comparison_plot")
    ##plt.show()

    roc_auc_score(y_train_5, y_scores_forest)

    y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
    print("line = 251",precision_score(y_train_5, y_train_pred_forest))
    print("line = 252",recall_score(y_train_5, y_train_pred_forest))

## 多类分类

    ## SGDClassifier
    sgd_clf.fit(X_train, y_train)
    sgd_clf.predict([some_digit])

    some_digit_scores = sgd_clf.decision_function([some_digit])
    print("line = 261", some_digit_scores)

    print("line = 263",np.argmax(some_digit_scores))
    print("line = 264", sgd_clf.classes_)
    print("line = 265", sgd_clf.classes_[5])

    ## SGDClassifier使用OvO策略创建多分类
    ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
    ovo_clf.fit(X_train, y_train)
    ovo_clf.predict([some_digit])
    print("line = 272", len(ovo_clf.estimators_))

    ## 训练一个 RandomForestClassifier
    forest_clf.fit(X_train, y_train)
    forest_clf.predict([some_digit])

    ## predict_proba 获取分类器为每个类分配给每个实例的概率
    forest_clf.predict_proba([some_digit])

    cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

## 误差分析
    ##  查看混淆矩阵  先使用cross_val_predict 进行预测  然后调用 confusion_matrix
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print("line = 292", conf_mx)

    plt.matshow(conf_mx, cmap=plt.cm.gray)
    save_fig("confusion_matrix_plot", tight_lay=False)
    plt.show()


    ##
    cl_a, cl_b = 3, 5
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

    plt.figure(figsize=(8,8))
    plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
    plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
    save_fig("error_analysis_digits_plot")
    plt.show()


##  多标签分类
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)

    knn_clf.predict([some_digit])

    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
    f1_score(y_multilabel, y_train_knn_pred, average="macro")

##  多输出分类
    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train
    y_test_mod = X_test

    some_index = 5500
    plt.subplot(121); plot_digit(X_test_mod[some_index])
    plt.subplot(122); plot_digit(y_test_mod[some_index])
    save_fig("noisy_digit_example_plot")
    plt.show()

    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_test_mod[some_index]])
    plot_digit(clean_digit)
    save_fig("cleaned_digit_example_plot")
    plt.show()


