from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
# 让笔记全程输入稳定
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error





# To plot pretty figures
# 导入绘图工具
##%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
# 设定图片保存路径，这里写了一个函数，后面直接调用即可
##PROJECT_ROOT_DIR = "F:\ML\Machine learning\Hands-on machine learning with scikit-learn and tensorflow"
PROJECT_ROOT_DIR =  os.path.dirname(os.path.realpath(__file__))
CHAPTER_ID = "07_Ensemble Learning and Random Forests"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
# 忽略无用警告
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")



def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.hot,
               interpolation="nearest")
    plt.axis("off")

def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)


if __name__ == '__main__':
    heads_proba = 0.51
    coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
    cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)

    plt.figure(figsize=(8, 3.5))
    plt.plot(cumulative_heads_ratio)
    plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
    plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
    plt.xlabel("Number of coin tosses")
    plt.ylabel("Heads ratio")
    plt.legend(loc="lower right")
    plt.axis([0, 10000, 0.42, 0.58])
    save_fig("law_of_large_numbers_plot")
    plt.show()


    X,y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard')

    voting_clf.fit(X_train, y_train)

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(probability=True, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='soft')

    voting_clf.fit(X_train, y_train)

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42),  # 基类预测器
        n_estimators=500,
        max_samples=100,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred_tree))

    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plot_decision_boundary(tree_clf, X, y)
    plt.title("Decision Tree", fontsize=14)
    plt.subplot(122)
    plot_decision_boundary(bag_clf, X, y)
    plt.title("Decision Trees with Bagging", fontsize=14)
    save_fig("decision_tree_without_and_with_bagging_plot")
    plt.show()

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42),  # 基类预测器
        n_estimators=500,
        bootstrap=True,
        n_jobs=-1,
        oob_score=True,
        random_state=40
    )

    bag_clf.fit(X_train, y_train)
    bag_clf.oob_score_

    y_pred = bag_clf.predict(X_test)
    accuracy_score(y_test, y_pred)

    print("line = 166", bag_clf.oob_decision_function_)

    rdn_clf = RandomForestClassifier
    rnd_clf.fit(X_train, y_train)

    y_pred_rf = rnd_clf.predict(X_test)

    np.sum(y_pred == y_pred_rf) / len(y_pred)  # almost identical predictions

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
        n_estimators=500,
        max_samples=1.0,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)

    iris = load_iris()
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rnd_clf.fit(iris["data"], iris["target"])
    for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)


    print("line = 195", rnd_clf.feature_importances_)

    plt.figure(figsize=(6, 4))

    for i in range(15):
        tree_clf = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
        indices_with_replacement = np.random.randint(0, len(X_train), len(X_train))
        tree_clf.fit(X[indices_with_replacement], y[indices_with_replacement])
        plot_decision_boundary(tree_clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.02, contour=False)

    plt.show()

    iris = load_iris()
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    rnd_clf.fit(iris["data"], iris["target"])
    for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)

    mnist = fetch_mldata('MNIST original', data_home='/home/inzahgi/test/jupyter/hand_on_ML/Hands-on-Machine-Learning/datasets')

    rnd_clf = RandomForestClassifier(random_state=42)
    rnd_clf.fit(mnist["data"], mnist["target"])

    plot_digit(rnd_clf.feature_importances_)

    cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
    cbar.ax.set_yticklabels(['Not important', 'Very important'])

    save_fig("mnist_feature_importance_plot")
    plt.show()

    ####  adaboost

    m = len(X_train)

    plt.figure(figsize=(11, 4))
    for subplot, learning_rate in ((121, 1), (122, 0.5)):
        sample_weights = np.ones(m)
        plt.subplot(subplot)
        for i in range(5):
            svm_clf = SVC(kernel="rbf", C=0.05, random_state=42)
            svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
            y_pred = svm_clf.predict(X_train)
            sample_weights[y_pred != y_train] *= (1 + learning_rate)
            plot_decision_boundary(svm_clf, X, y, alpha=0.2)
            plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
        if subplot == 121:
            plt.text(-0.7, -0.65, "1", fontsize=14)
            plt.text(-0.6, -0.10, "2", fontsize=14)
            plt.text(-0.5, 0.10, "3", fontsize=14)
            plt.text(-0.4, 0.55, "4", fontsize=14)
            plt.text(-0.3, 0.90, "5", fontsize=14)

    save_fig("boosting_plot")
    plt.show()

    list(m for m in dir(ada_clf) if not m.startswith("_") and m.endswith("_"))

    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1),
        n_estimators=200,
        algorithm="SAMME.R",
        learning_rate=0.5,
        random_state=42)

    ada_clf.fit(X_train, y_train)

    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

    tree_reg1 = DecisionTreeRegressor(max_depth=2)
    tree_reg1.fit(X, y)

    y2 = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=2)
    tree_reg2.fit(X, y2)

    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg3.fit(X, y3)

    X_new = np.array([[0.8]])

    y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    print("line = 291", y_pred)

    plt.figure(figsize=(11, 11))

    plt.subplot(321)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-",
                     data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Residuals and tree predictions", fontsize=16)

    plt.subplot(322)
    plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$",
                     data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Ensemble predictions", fontsize=16)

    plt.subplot(323)
    plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+",
                     data_label="Residuals")
    plt.ylabel("$y - h_1(x_1)$", fontsize=16)

    plt.subplot(324)
    plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
    plt.ylabel("$y$", fontsize=16, rotation=0)

    plt.subplot(325)
    plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
    plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
    plt.xlabel("$x_1$", fontsize=16)

    plt.subplot(326)
    plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8],
                     label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$y$", fontsize=16, rotation=0)

    save_fig("gradient_boosting_plot")
    plt.show()

    gbrt = GradientBoostingRegressor(
        max_depth=2,
        n_estimators=3,
        learning_rate=1.0,
        random_state=42
    )

    gbrt.fit(X, y)

    gbrt_slow_1 = GradientBoostingRegressor(
        max_depth=2,
        n_estimators=3,
        learning_rate=0.1,
        random_state=42
    )

    gbrt_slow_1.fit(X, y)

    gbrt_slow_2 = GradientBoostingRegressor(
        max_depth=2,
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    )

    gbrt_slow_2.fit(X, y)

    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plot_predictions([gbrt_slow_1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
    plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)

    plt.subplot(122)
    plot_predictions([gbrt_slow_2], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)

    save_fig("gbrt_learning_rate_plot")
    plt.show()

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
    gbrt.fit(X_train, y_train)

    errors = [mean_squared_error(y_val, y_pred)
              for y_pred in gbrt.staged_predict(X_val)]

    bst_n_estimators = np.argmin(errors)

    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    gbrt_best.fit(X_train, y_train)

    min_error = np.min(errors)
    print(min_error)

    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.plot(errors, "b.-")
    plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
    plt.plot([0, 120], [min_error, min_error], "k--")
    plt.plot(bst_n_estimators, min_error, "ko")
    plt.text(bst_n_estimators, min_error * 1.2, "Minimum", ha="center", fontsize=14)
    plt.axis([0, 120, 0, 0.01])
    plt.xlabel("Number of trees")
    plt.title("Validation error", fontsize=14)

    plt.subplot(122)
    plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)

    save_fig("early_stopping_gbrt_plot")
    plt.show()

    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break  # early stopping


    print("line = 436", gbrt.n_estimators)
    print("Minimum validation MSE:", min_error)

    try:
        import xgboost
    except ImportError as ex:
        print("Error: the xgboost library is not installed.")
        xgboost = None

    if xgboost is not None:  # not shown in the book
        xgb_reg = xgboost.XGBRegressor(random_state=42)
        xgb_reg.fit(X_train, y_train)

        y_pred = xgb_reg.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        print("Validation MSE:", val_error)

    if xgboost is not None:  # not shown in the book
        xgb_reg.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=2
                    )
        y_pred = xgb_reg.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        print("Validation MSE:", val_error)

    xgboost.XGBRegressor().fit(X_train, y_train) if xgboost is not None else None

    GradientBoostingRegressor().fit(X_train, y_train)


