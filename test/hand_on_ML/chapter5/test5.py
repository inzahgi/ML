
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import time

# to make this notebook's output stable across runs
# 让笔记全程输入稳定
np.random.seed(42)


from sklearn.svm import SVC
from sklearn import datasets
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier



from mpl_toolkits.mplot3d import Axes3D
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
PROJECT_ROOT_DIR =  os.path.dirname(os.path.realpath(__file__))
CHAPTER_ID = "05_Support Vector Machines"
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

## 自定义  线性核svm
class MyLinearSVC(BaseEstimator):
    def __init__(self, C=1, eta0=1, eta_d=10000, n_epochs=1000, random_state=None):
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.eta_d = eta_d

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)

    def fit(self, X, y):
        # Random initialization
        if self.random_state:
            np.random.seed(self.random_state)
        w = np.random.randn(X.shape[1], 1)  # n feature weights
        b = 0

        m = len(X)
        t = y * 2 - 1  # -1 if t==0, +1 if t==1
        X_t = X * t
        self.Js = []

        # Training
        for epoch in range(self.n_epochs):
            support_vectors_idx = (X_t.dot(w) + t * b < 1).ravel()
            X_t_sv = X_t[support_vectors_idx]
            t_sv = t[support_vectors_idx]

            J = 1 / 2 * np.sum(w * w) + self.C * (np.sum(1 - X_t_sv.dot(w)) - b * np.sum(t_sv))
            self.Js.append(J)

            w_gradient_vector = w - self.C * np.sum(X_t_sv, axis=0).reshape(-1, 1)
            b_derivative = -C * np.sum(t_sv)

            w = w - self.eta(epoch) * w_gradient_vector
            b = b - self.eta(epoch) * b_derivative

        self.intercept_ = np.array([b])
        self.coef_ = np.array([w])
        support_vectors_idx = (X_t.dot(w) + t * b < 1).ravel()
        self.support_vectors_ = X[support_vectors_idx]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)


## 绘制决策边界
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0] ## 获取 间隔斜率
    b = svm_clf.intercept_[0]  ##  间隔截距

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)  ##  画出输入范围
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]  ## 画出决策边界
    ## 上下软间隔
    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    ##  画出支持向量和决策边界
    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA') ##  scatter画出散点图
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

##  画出预测图
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)  ##  生成网格坐标
    X = np.c_[x0.ravel(), x1.ravel()]  ##  生成坐标对
    y_pred = clf.predict(X).reshape(x0.shape)  ##  生成预测结果
    y_decision = clf.decision_function(X).reshape(x0.shape)   ## 生成判决
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)  ## 画出不同预测色块
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)  ##  画出不同判决边界等高线


def gaussian_rbf(x, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(x - landmark, axis=1)**2)


    ##  找出支撑向量
def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X) ## 生成预测值
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon) ##  计算间隔误差
    return np.argwhere(off_margin)

##  画出svm回归
def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)
##  画出3D决策边界
def plot_3D_decision_function(ax, w, b, x1_lim=[4, 6], x2_lim=[0.8, 2.8]):
    x1_in_bounds = (X[:, 0] > x1_lim[0]) & (X[:, 0] < x1_lim[1])
    X_crop = X[x1_in_bounds]
    y_crop = y[x1_in_bounds]
    x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
    x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
    x1, x2 = np.meshgrid(x1s, x2s)
    xs = np.c_[x1.ravel(), x2.ravel()]
    df = (xs.dot(w) + b).reshape(x1.shape)
    m = 1 / np.linalg.norm(w)
    boundary_x2s = -x1s*(w[0]/w[1])-b/w[1]
    margin_x2s_1 = -x1s*(w[0]/w[1])-(b-1)/w[1]
    margin_x2s_2 = -x1s*(w[0]/w[1])-(b+1)/w[1]
    ax.plot_surface(x1s, x2, np.zeros_like(x1),
                    color="b", alpha=0.2, cstride=100, rstride=100)
    ax.plot(x1s, boundary_x2s, 0, "k-", linewidth=2, label=r"$h=0$")
    ax.plot(x1s, margin_x2s_1, 0, "k--", linewidth=2, label=r"$h=\pm 1$")
    ax.plot(x1s, margin_x2s_2, 0, "k--", linewidth=2)
    ax.plot(X_crop[:, 0][y_crop==1], X_crop[:, 1][y_crop==1], 0, "g^")
    ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
    ax.plot(X_crop[:, 0][y_crop==0], X_crop[:, 1][y_crop==0], 0, "bs")
    ax.axis(x1_lim + x2_lim)
    ax.text(4.5, 2.5, 3.8, "Decision function $h$", fontsize=15)
    ax.set_xlabel(r"Petal length", fontsize=15)
    ax.set_ylabel(r"Petal width", fontsize=15)
    ax.set_zlabel(r"$h = \mathbf{w}^T \mathbf{x} + b$", fontsize=18)
    ax.legend(loc="upper left", fontsize=16)

##  画出2D决策边界
def plot_2D_decision_function(w, b, ylabel=True, x1_lim=[-3, 3]):
    x1 = np.linspace(x1_lim[0], x1_lim[1], 200)
    y = w * x1 + b
    m = 1 / w

    plt.plot(x1, y)
    plt.plot(x1_lim, [1, 1], "k:")
    plt.plot(x1_lim, [-1, -1], "k:")
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot([m, m], [0, 1], "k--")
    plt.plot([-m, -m], [0, -1], "k--")
    plt.plot([-m, m], [0, 0], "k-o", linewidth=3)
    plt.axis(x1_lim + [-2, 2])
    plt.xlabel(r"$x_1$", fontsize=16)
    if ylabel:
        plt.ylabel(r"$w_1 x_1$  ", rotation=0, fontsize=16)
    plt.title(r"$w_1 = {}$".format(w), fontsize=16)

if __name__ == '__main__':
    ##  获取iris 数据集
    iris = datasets.load_iris()
    ##  获取第三列 和第四列
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    ##  目标结果
    y = iris["target"]

    ##将结果转换为 布尔值
    setosa_or_versicolor = (y == 0) | (y == 1)
    ##  ??? 并没什么用
    X = X[setosa_or_versicolor]
    y = y[setosa_or_versicolor]

    # SVM Classifier model
    svm_clf = SVC(kernel="linear", C=float("inf"))
    svm_clf.fit(X, y)

    # Bad models
    x0 = np.linspace(0, 5.5, 200)  ##  训练数据
    ##  假设的预测分隔
    pred_1 = 5 * x0 - 20
    pred_2 = x0 - 1.8
    pred_3 = 0.1 * x0 + 0.5

    ##  画出错误模型的决策边界
    plt.figure(figsize=(12, 2.7))

    plt.subplot(121)
    plt.plot(x0, pred_1, "g--", linewidth=2)
    plt.plot(x0, pred_2, "m-", linewidth=2)
    plt.plot(x0, pred_3, "r-", linewidth=2)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.subplot(122)
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo")
    plt.xlabel("Petal length", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    save_fig("large_margin_classification_plot")
    plt.show()

    ##  利用svm训练
    iris=datasets.load_iris()
    X = iris["data"][:,(2,3)] # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64) # Iris-Viginica

    svm_clf = Pipeline((
        ("scaler",StandardScaler()),
        ("linear_svc",LinearSVC(C=1,loss="hinge")),
    ))

    svm_clf.fit(X,y)

    print("line = 283 svm_clf.predict([[5.5, 1.7]]) = {}".format(svm_clf.predict([[5.5, 1.7]])))
    ## 比较 C 参数的影响   和软间隔大小成反比
    scaler = StandardScaler()
    svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
    svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

    scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])

    scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])

    scaled_svm_clf1.fit(X, y)
    scaled_svm_clf2.fit(X, y)

    # Convert to unscaled parameters
    b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
    b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
    w1 = svm_clf1.coef_[0] / scaler.scale_
    w2 = svm_clf2.coef_[0] / scaler.scale_
    svm_clf1.intercept_ = np.array([b1])
    svm_clf2.intercept_ = np.array([b2])
    svm_clf1.coef_ = np.array([w1])
    svm_clf2.coef_ = np.array([w2])

    # Find support vectors (LinearSVC does not do this automatically)
    t = y * 2 - 1
    support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
    support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
    svm_clf1.support_vectors_ = X[support_vectors_idx1]
    svm_clf2.support_vectors_ = X[support_vectors_idx2]
    ##  画出c=1 时 决策边界和支撑向量
    plt.figure(figsize=(12, 3.2))
    plt.subplot(121)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="Iris-Virginica")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs", label="Iris-Versicolor")
    plot_svc_decision_boundary(svm_clf1, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
    plt.axis([4, 6, 0.8, 2.8])
    ##  画出c=100 时 决策边界和支撑向量
    plt.subplot(122)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plot_svc_decision_boundary(svm_clf2, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
    plt.axis([4, 6, 0.8, 2.8])

    save_fig("regularization_plot")

##  非线性分类
    X1D = np.linspace(-4, 4, 9).reshape(-1, 1)  ##  生成9个元素的列矩阵
    X2D = np.c_[X1D, X1D ** 2]  ## 生成 二维特征矩阵
    y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])  ##  目标值

    plt.figure(figsize=(11, 4))
    ##  画出一维的数据分布图
    plt.subplot(121)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.plot(X1D[:, 0][y == 0], np.zeros(4), "bs")
    plt.plot(X1D[:, 0][y == 1], np.zeros(5), "g^")
    plt.gca().get_yaxis().set_ticks([])
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.axis([-4.5, 4.5, -0.2, 0.2])
    ##  画出 二维的分布图
    plt.subplot(122)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot(X2D[:, 0][y == 0], X2D[:, 1][y == 0], "bs")
    plt.plot(X2D[:, 0][y == 1], X2D[:, 1][y == 1], "g^")
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
    plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
    plt.axis([-4.5, 4.5, -1, 17])

    plt.subplots_adjust(right=1)
    plt.title('Figure 5-5. Adding features to make a dataset linearly separable')  # not shown in the book
    save_fig("higher_dimensions_plot", tight_layout=False)
    plt.show()

    ##  make_moons 生成办环型数据
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
    ##  画出图像
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.show()
    ##  多阶特征 svm训练
    polynomial_svm_clf = Pipeline((("poly_features", PolynomialFeatures(degree=3)),
                                   ("scaler", StandardScaler()),
                                   ("svm_clf", LinearSVC(C=10, loss="hinge"))
                                   ))

    polynomial_svm_clf.fit(X, y)
    ##  画出多阶的情况
    plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title('Figure 5-6. Linear SVM classifier using polynomial features')  # not shown in the book

    save_fig("moons_polynomial_svc_plot")
    plt.show()
    ##  不同维度的参数拟合
    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])

    poly_kernel_svm_clf.fit(X, y)

    poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])

    poly100_kernel_svm_clf.fit(X, y)

    plt.figure(figsize=(11, 4))
    ##  画出c=5时的判决情况
    plt.subplot(121)
    plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=3, r=1, C=5$", fontsize=18)
    ## 画出c=100时的判决情况
    plt.subplot(122)
    plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=10, r=100, C=5$", fontsize=18)

    save_fig("moons_kernelized_polynomial_svc_plot")
    plt.show()

    ##  高斯核 方法
    gamma = 0.3

    x1s = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)
    x2s = gaussian_rbf(x1s, -2, gamma)
    x3s = gaussian_rbf(x1s, 1, gamma)

    XK = np.c_[gaussian_rbf(X1D, -2, gamma), gaussian_rbf(X1D, 1, gamma)]
    yk = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.scatter(x=[-2, 1], y=[0, 0], s=150, alpha=0.5, c="red")
    plt.plot(X1D[:, 0][yk == 0], np.zeros(4), "bs")  ##  画出在x轴上的负例
    plt.plot(X1D[:, 0][yk == 1], np.zeros(5), "g^")  ##  画出在y轴上的正例
    plt.plot(x1s, x2s, "g--")  ## 画出高斯函数
    plt.plot(x1s, x3s, "b:")
    plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"Similarity", fontsize=14)
    plt.annotate(r'$\mathbf{x}$',
                 xy=(X1D[3, 0], 0),
                 xytext=(-0.5, 0.20),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=18,
                 )
    plt.text(-2, 0.9, "$x_2$", ha="center", fontsize=20)
    plt.text(1, 0.9, "$x_3$", ha="center", fontsize=20)
    plt.axis([-4.5, 4.5, -0.1, 1.1])

    plt.subplot(122)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot(XK[:, 0][yk == 0], XK[:, 1][yk == 0], "bs") ##  画出负例点
    plt.plot(XK[:, 0][yk == 1], XK[:, 1][yk == 1], "g^") ##  画出正例点
    plt.xlabel(r"$x_2$", fontsize=20)
    plt.ylabel(r"$x_3$  ", fontsize=20, rotation=0)
    plt.annotate(r'$\phi\left(\mathbf{x}\right)$',
                 xy=(XK[3, 0], XK[3, 1]),
                 xytext=(0.65, 0.50),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=18,
                 )
    plt.plot([-0.1, 1.1], [0.57, -0.1], "r--", linewidth=3)  ##  画出认为的决策边界
    plt.axis([-0.1, 1.1, -0.1, 1.1])

    plt.subplots_adjust(right=1)

    save_fig("kernel_method_plot")
    plt.show()
    ## 高斯函数将一维特征转换为二维特征
    x1_example = X1D[3, 0]
    for landmark in (-2, 1):
        k = gaussian_rbf(np.array([[x1_example]]), np.array([[landmark]]), gamma)
        print("line = 482 Phi({}, {}) = {}".format(x1_example, landmark, k))
    ##  rbf核函数训练
    rbf_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ))

    rbf_kernel_svm_clf.fit(X, y)
    ## 生成参数网格
    gamma1, gamma2 = 0.1, 5
    C1, C2 = 0.001, 1000
    hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)
    ##  不同gamma 和 C值的训练
    svm_clfs = []
    for gamma, C in hyperparams:
        rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
        rbf_kernel_svm_clf.fit(X, y)
        svm_clfs.append(rbf_kernel_svm_clf)

    plt.figure(figsize=(11, 7))
    ##  enumerate 返回带序号的枚举对象  画出不同参数的训练结果
    for i, svm_clf in enumerate(svm_clfs):
        plt.subplot(221 + i)
        plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
        plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
        gamma, C = hyperparams[i]
        plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)

    save_fig("moons_rbf_svc_plot")
    plt.show()

    ##  生成训练数据
    np.random.seed(42)
    m = 50
    X = 2 * np.random.rand(m, 1)
    y = (4 + 3 * X + np.random.randn(m, 1)).ravel()
    ## 线性svm 训练
    svm_reg = LinearSVR(epsilon=1.5, random_state=42)
    svm_reg.fit(X, y)
    ##  线性svm  用不同epsilon (不敏感度)  训练
    svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)
    svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)
    svm_reg1.fit(X, y)
    svm_reg2.fit(X, y)
    ##  找出支撑向量
    svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
    svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)
    ##  svm回归 预测结果
    eps_x1 = 1
    eps_y_pred = svm_reg1.predict([[eps_x1]])
    ##  画出不同epsilon 的结果
    plt.figure(figsize=(9, 4))
    plt.subplot(121)  ##  epsilon=1.5的训练图
    plot_svm_regression(svm_reg1, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$".format(svm_reg1.epsilon), fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    # plt.plot([eps_x1, eps_x1], [eps_y_pred, eps_y_pred - svm_reg1.epsilon], "k-", linewidth=2)
    plt.annotate(
        '', xy=(eps_x1, eps_y_pred), xycoords='data',
        xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
    )
    plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
    plt.subplot(122) ## epsilon=0.5 的训练图
    plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)
    save_fig("svm_regression_plot")
    plt.show()
##  多项式特征 不同C值的训练拟合情况
    np.random.seed(42)
    m = 100
    X = 2 * np.random.rand(m, 1) - 1
    y = (0.2 + 0.1 * X + 0.5 * X ** 2 + np.random.randn(m, 1) / 10).ravel()

    svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg.fit(X, y)

    svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
    svm_poly_reg1.fit(X, y)
    svm_poly_reg2.fit(X, y)

    plt.figure(figsize=(9, 4))
    plt.subplot(121) ## C =100
    plot_svm_regression(svm_poly_reg1, X, y, [-1, 1, 0, 1])
    plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg1.degree, svm_poly_reg1.C, svm_poly_reg1.epsilon),
              fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.subplot(122) ## C = 0.01
    plot_svm_regression(svm_poly_reg2, X, y, [-1, 1, 0, 1])
    plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg2.degree, svm_poly_reg2.C, svm_poly_reg2.epsilon),
              fontsize=18)
    save_fig("svm_with_polynomial_kernel_plot")
    plt.show()
    ###  导入花瓣数据
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica
    ## 画出花瓣数据的3D特征图
    fig = plt.figure(figsize=(11, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    plot_3D_decision_function(ax1, w=svm_clf2.coef_[0], b=svm_clf2.intercept_[0])
    plt.title('Figure 5-12. Decision function for the iris dataset')

    save_fig("iris_3D_plot")
    plt.show()

    plt.figure(figsize=(12, 3.2))

    plt.subplot(121) # w = 1
    plot_2D_decision_function(1, 0)

    plt.subplot(122) # w = 0.5
    plot_2D_decision_function(0.5, 0, ylabel=False)

    save_fig("small_w_large_margin_plot")
    plt.show()

    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

    svm_clf = SVC(kernel="linear", C=1)
    svm_clf.fit(X, y)
    print('line = 609  svm_clf.predict([[5.3, 1.3]]) = {}'.format(svm_clf.predict([[5.3, 1.3]])));

    t = np.linspace(-2, 4, 200)
    h = np.where(1 - t < 0, 0, 1 - t)  # max(0, 1-t)
    ##  画出 hinge 损失函数
    plt.figure(figsize=(5, 2.8))
    plt.plot(t, h, "b-", linewidth=2, label="$max(0, 1 - t)$")
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.yticks(np.arange(-1, 2.5, 1))
    plt.xlabel("$t$", fontsize=16)
    plt.axis([-2, 4, -1, 2.5])
    plt.legend(loc="upper right", fontsize=16)
    plt.title('Hinge Loss')

    save_fig("hinge_plot")
    plt.show()
    ##  make_moons 生成半环数据
    X, y = make_moons(n_samples=1000, noise=0.4, random_state=42)
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    ##  早期停止训练法
    tol = 0.1
    tols = []
    times = []
    for i in range(10):
        svm_clf = SVC(kernel="poly", gamma=3, C=10, tol=tol, verbose=1)
        t1 = time.time()
        svm_clf.fit(X, y)
        t2 = time.time()
        times.append(t2 - t1)
        tols.append(tol)
        print(i, tol, t2 - t1)
        tol /= 10
    plt.semilogx(tols, times) ##  画出 时间和停止误差
##  自定义实现svm算法
    # Training set
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)  # Iris-Virginica

    C = 2
    svm_clf = MyLinearSVC(C=C, eta0=10, eta_d=1000, n_epochs=60000, random_state=2)
    svm_clf.fit(X, y)
    svm_clf.predict(np.array([[5, 2], [4, 1]]))

    plt.plot(range(svm_clf.n_epochs), svm_clf.Js)
    plt.axis([0, svm_clf.n_epochs, 0, 100])

    print("line = 649", svm_clf.intercept_, svm_clf.coef_)

    svm_clf2 = SVC(kernel="linear", C=C)
    svm_clf2.fit(X, y.ravel())
    print(svm_clf2.intercept_, svm_clf2.coef_)

    yr = y.ravel()
    plt.figure(figsize=(12, 3.2))
    plt.subplot(121)
    plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^", label="Iris-Virginica")
    plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs", label="Not Iris-Virginica")
    plot_svc_decision_boundary(svm_clf, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.title("MyLinearSVC", fontsize=14)
    plt.axis([4, 6, 0.8, 2.8])

    plt.subplot(122)
    plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
    plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
    plot_svc_decision_boundary(svm_clf2, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.title("SVC", fontsize=14)
    plt.axis([4, 6, 0.8, 2.8])

    sgd_clf = SGDClassifier(loss="hinge", alpha=0.017, max_iter=50, random_state=42)
    sgd_clf.fit(X, y.ravel())

    m = len(X)
    t = y * 2 - 1  # -1 if t==0, +1 if t==1
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias input x0=1
    X_b_t = X_b * t
    sgd_theta = np.r_[sgd_clf.intercept_[0], sgd_clf.coef_[0]]
    print(sgd_theta)
    support_vectors_idx = (X_b_t.dot(sgd_theta) < 1).ravel()
    sgd_clf.support_vectors_ = X[support_vectors_idx]
    sgd_clf.C = C

    plt.figure(figsize=(5.5, 3.2))
    plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
    plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
    plot_svc_decision_boundary(sgd_clf, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.title("SGDClassifier", fontsize=14)
    plt.axis([4, 6, 0.8, 2.8])