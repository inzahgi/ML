
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd

import os

np.random.seed(42)


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from sklearn import datasets
from sklearn.linear_model import LogisticRegression




import  matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CHAPTER_ID = "04_Training Models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "..\\images", CHAPTER_ID)

def save_fig(fig_id, tight_lay=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    #print("line 42 path = ", path)
    print("Saving figure", fig_id)
    if tight_lay:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


##  画出梯度下降图
def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)  ##求预测值
            style = "b-" if iteration > 0 else "r--"  ##  迭代次数剩余大于0时为蓝色实线 否则为红色虚线
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)  ##  求当前梯度
        theta = theta - eta * gradients ## 更新 theta
        if theta_path is not None:
            theta_path.append(theta) ##  保存theta
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

##  按照学习计划  跟新学习率
t0, t1 = 5, 50
def learning_schedule(t):
    return t0/(t+t1)

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10) ## 拆分训练集和测试集
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])  ## 训练样本
        y_train_predict = model.predict(X_train[:m]) ## 预测训练集结果
        y_val_predict = model.predict(X_val)  ##  预测测试集结果
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))   ## 计算训练误差
        val_errors.append(mean_squared_error(y_val, y_val_predict))   ##  计算预测误差
    ##  绘制误差曲线
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)

## 画出 线性曲线拟合情况
def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b--", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:   ## 如果是高阶拟合 设置参数
            model = Pipeline([
                ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                ("std_scaler", StandardScaler()),
                ("regul_reg", model),
            ])
        model.fit(X, y) ## 训练模型
        y_new_regul= model.predict(X_new) ## 预测结果
        lw = 2 if alpha > 0 else 1  ##  正则参数 alpha 大于0时设置为2  默认为1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))  ##  画出拟合曲线

    plt.plot(X, y, "b.", linewidth=3) ##  画出训练数据点
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])


def bgd_path(theta, X, y, l1, l2, core=1, eta=0.1, n_iterations = 50):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = core * 2/len(X) * X.T.dot((theta) - y) + l1 * np.sign(theta) + 2 * l2 * theta
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)


if __name__ == '__main__':
    ## 生成随机数据
    X=2*np.random.rand(100, 1)
    y=4+3*X+np.random.randn(100, 1)
    ##  画出随机数据的图像
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    save_fig("generated_plot")
    plt.show()
    ## 利用求逆矩阵 求得斜率
    X_b = np.c_[np.ones((100, 1)), X]##  偏置项
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print("line = 134 theta_best = \t", theta_best)
    ##  利用求得的theta 进行预测
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict=X_new_b.dot(theta_best)
    print("line = 139 y_predict = \t", y_predict)
    ##  画出预测结果
    plt.plot(X_new, y_predict, "r--")
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.show()

    ## 利用scikit-learn 训练线性模型
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print("line = 150 lin_reg = \t", lin_reg.intercept_, lin_reg.coef_)
    print("line = 151 lin_reg.predict = \t", lin_reg.predict(X_new))
    ## LinearRegression类基于scipy.linalg.lstsq()函数(名称代表'最小二乘')，直接调用它
    theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
    print("line = 154 theta_best_svd = \t", theta_best_svd)


   ##  批量梯度下降
    eta = 0.1   # 学习率
    n_iterations=1000  ## 迭代次数
    m=100  ## 个数
    theta=np.random.randn(2, 1)  ##  初始值 随机数
    ##  利用梯度下降算法 迭代训练 theta
    for interation in range(n_iterations):
        gradients=2/m*X_b.T.dot(X_b.dot(theta)-y)
        theta = theta - eta * gradients

    print("line = 167 theta = \t", theta)

    print("line = 169  predict = \t", X_new_b.dot(theta))

    theta_path_bgd = []

    np.random.seed(42)
    theta = np.random.randn(2, 1)
    ##  绘制 学习率 太小 合适 太大 的训练情况
    plt.figure(figsize=(10, 4))
    plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
    plt.subplot(133); plot_gradient_descent(theta, eta=0.5)

    save_fig("gradient_descent_plot")
    plt.show()

    theta_path_sgd = []
    m = len(X_b)
    np.random.seed(42)

    n_epochs=50
    theta = np.random.randn(2, 1)
    ## 循环迭代
    for epoch in range(n_epochs):
        for i in range(m):## 对每一个输入
            if epoch == 0 and i < 20:  ## 当进行最后一次的迭代时候 输入训练数据小于20 时 绘制预测线性回归图像
                y_predict = X_new_b.dot(theta)
                style = "b--" if i > 0 else "r--"
                plt.plot(X_new, y_predict, style)
            ## 随机数 作索引
            random_index = np.random.randint(m)
            ## 获取前后 训练数据
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            ## 训练 梯度下降算法
            gradients = 2*xi.T.dot(xi.dot(theta)-yi)
            eta=learning_schedule(epoch*m+i)
            theta=theta-eta*gradients
            theta_path_sgd.append(theta)
    ## 画出梯度收敛
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    save_fig("sgd_plot")
    plt.show()
    ##  最后收敛的 theta值
    print("line = 218", theta)
    ##  使用随机梯度下降算法 训练
    sgd_reg=SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
    sgd_reg.fit(X, y.ravel())

    print("line = 223 sgd_reg.intercept = \t", sgd_reg.intercept_, "sgd_reg_coef = \t", sgd_reg.coef_)

##  小批量梯度训练
    theta_path_mgd = []
    n_iterations = 50
    minibatch_size = 20

    np.random.seed(42)
    theta = np.random.randn(2, 1)  ##  theta 初始值

    t0,t1 = 200, 1000
    t=0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m) ##  生成伪随机数组  设置随机种子
        X_b_shuffled = X_b[shuffled_indices]  ##  打散训练数据
        y_shuffled = y[shuffled_indices]  ##  打散训练数据标注结果
        for i in range(0, m, minibatch_size):  ##  小批次迭代循环
            t += 1
            xi = X_b_shuffled[i:i+minibatch_size] ## 取新的小批次
            yi = y_shuffled[i:i+minibatch_size]
            gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)  ##  求当前梯度
            eta = learning_schedule(t) ##  跟新学习速率
            theta = theta - eta * gradients ##  更新theta
            theta_path_mgd.append(theta)  ##  保存theta

    print("line = 248 theta = \t", theta)
    ##  批量梯度下降 随机梯度下降 小批量梯度下降的 训练过程
    theta_path_bgd = np.array(theta_path_bgd)
    theta_path_sgd = np.array(theta_path_sgd)
    theta_path_mgd = np.array(theta_path_mgd)
    ##  绘制三种梯度下降
    plt.figure(figsize=(7, 4))
    plt.plot(theta_path_sgd[:,0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
    plt.plot(theta_path_mgd[:,0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
    plt.plot(theta_path_bgd[:,0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
    plt.legend(loc="upper left", fontsize=16)
    plt.xlabel(r"$\theta_0$", fontsize=20)
    plt.ylabel(r"$\theta_1$", fontsize=20, rotation=0)
    plt.axis([2.5, 4.5, 2.3, 3.9])
    save_fig("gradient_descent_path_plot")
    plt.show()

    np.random.rand(42)
    m = 100
    ## 生成随机训练数据
    X = 6 * np.random.rand(m, 1) - 3
    ## 生成二次曲线 目标
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
    ## 画出随机数据图
    plt.plot(X, y, "b.")
    plt.title("Figure 4-12. Generated nonlinear and noisy dataset")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    save_fig("quadratic_data_plot")
    plt.show()
    ##  利用polynomialFeatures 转换训练数据 添加多维(当前2维) 数据
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly=poly_features.fit_transform(X)
    print("line = 282，X[0] = \t", X[0])
    print("line = 283, X_poly[0] = \t", X_poly[0])

    ##  线性回归训练
    lin_reg=LinearRegression()
    lin_reg.fit(X_poly, y)
    print("line = 288 lin_reg.intercept = \t",lin_reg.intercept_, "\tlin_reg.coef_ = \t",lin_reg.coef_)
    ##  画出二维训练结果 拟合图
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)  ##  预测结果
    plt.plot(X, y, "b.") ##  训练点
    plt.title("Figure 4-13. Polynomial Regression model predictions")
    plt.plot(X_new, y_new, "r--", linewidth=2, label="Predictions")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([-3, 3, 0, 10])
    save_fig("quadratic_predictions_plot")
    plt.show()

    ##  训练画出 300阶 2阶 1阶 的训练结果
    for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_regression = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
        polynomial_regression.fit(X, y)
        y_newbig = polynomial_regression.predict(X_new)
        plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)
    ##  画出训练数据
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([-3, 3, 0, 10])
    save_fig("high_degree_polynomials_plot")
    plt.show()

    ## 绘制模型的学习曲线
    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)
    plt.axis([0, 80, 0, 3])
    save_fig("underfitting_learning_curves_plot")
    plt.show()

    ##  10次多项式模型的学习曲线
    polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])
    ##  画出多阶回归拟合情况
    plot_learning_curves(polynomial_regression, X, y)
    plt.axis([0, 80, 0, 3])
    plt.title("Figure.4-16")
    save_fig("learning_curves_plot")
    plt.show()

    ## 岭回归
    np.random.seed(42)
    m = 20
    X = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * X + np.random.rand(m, 1)/ 1.5
    X_new = np.linspace(0, 3, 100).reshape(100, 1)


    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title('Figure 4-17. Ridge Regression')
    plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42) ## 画出一阶拟合曲线
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plt.title("Figure 4-17. Ridge Regression")
    plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42) ## 画出多阶拟合曲线

    save_fig("ridge_regression_plot")
    plt.show()

    ## 利用cholesky矩阵分解 得到闭式解
    ridge_reg = Ridge(alpha=1, solver="cholesky")
    ridge_reg.fit(X, y)
    ridge_reg.predict([[1.5]])
    ##  超参数 指定为L2 正则
    sgd_reg=SGDRegressor(penalty="l2")
    sgd_reg.fit(X, y.ravel())
    print("line = 370 sgd_reg.predict([[1.5]]) = \t", sgd_reg.predict([[1.5]]))

    ##  使用sgd  随机平均梯度算法最小化 有限和
    ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
    ridge_reg.fit(X, y)
    print("line = 375 ridge_reg.predict([[1.5]]) = \t", ridge_reg.predict([[1.5]]))
    ##  画出一阶和高阶 的 lasso 线性回归拟合曲线
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title('Figure 4-18. Lasso Regression')
    plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plt.title("Figure 4-18 Lasso Regression")
    plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)

    save_fig("lasso_regression_plot")
    plt.show()


    t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.5
    ##  生成训练数据
    t1s = np.linspace(t1a, t1b, 500)
    t2s = np.linspace(t1a, t2b, 500)
    t1, t2 = np.meshgrid(t1s, t2s)
    T = np.c_[t1.ravel(), t2.ravel()]
    Xr = np.array([[-1, 1], [-0.3, -1], [1, 0.1]])
    yr = 2 * Xr[:, :1] + 0.5 * Xr[:, 1:]

    J = (1/len(Xr) * np.sum((T.dot(Xr.T) - yr.T)**2, axis=1)).reshape(t1.shape)

    N1 = np.linalg.norm(T, ord=1, axis=1).reshape(t1.shape)
    N2 = np.linalg.norm(T, ord=2, axis=1).reshape(t1.shape)

    t_min_idx = np.unravel_index(np.argmin(J), J.shape)
    t1_min, t2_min = t1[t_min_idx], t2[t_min_idx]

    t_init = np.array([[0.25], [-1]])

#     plt.figure(figsize=(10, 6))
#     for i, N, l1, l2, title in ((0, N1, 0.5, 0, "Lasso"), (1, N2, 0, 0.1, "Ridge")):
#         JR = J + l1 * N1 + l2 * N2**2
#         tr_min_idx = np.unravel_index(np.argmin(JR), JR.shape)
#         tlr_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]
#
#         levelsJ = (np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(J)) + np.min(J)
#         levelsJR = (np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(JR) - np.min(JR)) + np.min(JR)
#         levelsN = np.linspace(0, np.max(N), 10)
#
#         path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
#         path_JR = bgd_path(t_init, Xr, yr, l1, l2)
#         path_N = bgd_path(t_init, Xr, yr, np.sign(l1)/3, np.sign(l2), core=0)
#
#         plt.subplot(221 + i * 2)
#         plt.grid(True)
#         plt.axhline(y=0, color='k')
#         plt.axvline(x=0, color='k')
#         plt.contour(t1, t2, J, levels=levelsJ, alpha=0.9)
#         plt.contour(t1, t2, N, levels=levelsN)
#         plt.plot(path_J[:, 0], path_J[:, 1], "w-o")
#         plt.plot(t1_min, t2_min, "rs")
#         plt.title(r"$\ell_{}$ penalty".format(i + 1), fontsize=16)
#         plt.axis([t1a, t1b, t2a, t2b])
#         if i == 1:
#             plt.xlabel(r"$\theta_1$", fontsize=20)
#         plt.ylabel(r"$\theta_2$", fontsize=20, rotation=0)
#
#         plt.subplot(222 + i * 2)
#         plt.grid(True)
#         plt.axhline(y=0, color='k')
#         plt.axvline(x=0, color='k')
#         plt.contour(t1, t2, JR, levels=levelsJR, alpha=0.9)
#         plt.plot(path_JR[:, 0], path_JR[:, 1], "w-o")
#         plt.plot(tlr_min, t2r_min, "rs")
#         plt.title(title, fontsize=16)
#         plt.axis([t1a, t1b, t2a, t2b])
#         if i == 1:
#             plt.xlabel(r"$\theta_1$", fontsize=20)
#
#     save_fig("lasso_vs_ridge_plot")
#     plt.show()
#
#     ##  使用Lasso 类的例子
#     lasso_reg = Lasso(alpha=0.1)
#     lasso_reg.fit(X, y)
#     lasso_reg.predict([[1.5]])
#
#     ##  elastic net
#     elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
#     elastic_net.fit(X, y)
#     elastic_net.predict([[1.5]])
#
#     ##  early stopping
#     np.random.seed(42)
#     m = 100
#     X = 6 * np.random.rand(m, 1) - 3
#     y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)
#
#     X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)
#     poly_scaler = Pipeline([
#         ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
#         ("std_scaler", StandardScaler()),
#     ])
#
#     X_train_poly_scaled = poly_scaler.fit_transform(X_train)
#     X_val_poly_scaled = poly_scaler.transform(X_val)
#
#     sgd_reg = SGDRegressor(max_iter= 1,
#                            penalty=None,
#                            eta0=0.0005,
#                            warm_start=True,
#                            learning_rate="constant",
#                            random_state=42)
#     n_epochs = 500
#     train_errors, val_errors = [], []
#     for epoch in range(n_epochs):
#         sgd_reg.fit(X_train_poly_scaled, y_train)
#         y_train_predict = sgd_reg.predict(X_train_poly_scaled)
#         y_val_predict = sgd_reg.predict(X_val_poly_scaled)
#         train_errors.append(mean_squared_error(y_train, y_train_predict))
#         val_errors.append(mean_squared_error(y_val, y_val_predict))
#
#     best_epoch = np.argmin(val_errors)
#     best_val_rmse = np.sqrt(val_errors[best_epoch])
#
#     plt.annotate('Best model',
#                  xy=(best_epoch, best_val_rmse),
#                  xytest=(best_epoch, best_val_rmse + 1),
#                  ha="center",
#                  arrowprops=dict(facecolor='black', shrink=0.005),
#                  fontsize=16,
#                  )
#     best_val_rmse -= 0.03
#     plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
#     plt.plot(np.sqrt(val_errors), "b--", linewidth=3, label="Validation set")
#     plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
#     plt.legend(loc="upper right", fontsize=14)
#     plt.xlabel("Epoch", fontsize=14)
#     plt.ylabel("RMSE", fontsize=14)
#     plt.title("Figure 4-20. Early stopping regularization")
#     save_fig("early_stopping_plot")
#     plt.show()
#
#     sgd_reg=SGDRegressor(max_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
#
#     minimum_val_error=float("inf")
#     best_epoch=None
#     best_model=None
#     for epoch in range(1000):
#         sgd_reg.fit(X_train_poly_scaled, y_train)
#         y_val_predict=sgd_reg.predict(X_val_poly_scaled)
#         val_error = mean_squared_error(y_val_predict, y_val)
#         if val_error < minimum_val_error:
#             minimum_val_error = val_error
#             best_epoch = epoch
#             best_model=clone(sgd_reg)
#
#     print("line = 520", best_epoch, best_model)
#
#
# ##  逻辑回归
#     ##   logistic function
#     t = np.linspace(-10, 10, 100)
#     sig = 1/ (1 + np.exp(-t))
#     plt.figure(figsize=(9, 3))
#     plt.plot([-10, 10], [0, 0], "k-")
#     plt.plot([-10, 10], [0.5, 0.5], "k:")
#     plt.plot([-10, 10], [1, 1], "k:")
#     plt.plot([0, 0], [-1.1, 1.1], "k-")
#     plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
#     plt.xlabel("t")
#     plt.legend(loc="upper left", fontsize=20)
#     plt.axis([-10, 10, -0.1, 1.1])
#     plt.title('Figure 4-21. Logistic function')
#     save_fig("logistic_function_plot")
#     plt.show()
#
#
#     ## 获取花瓣特征数据
#     iris=datasets.load_iris()
#     list(iris.keys())
#
#     print("lie = 546", iris.DESCR)
#     X = iris["data"][:,3:]
#     y=(iris["target"]==2)
#
#
#     log_reg=LogisticRegression()
#     log_reg.fit(X, y)
#
#     X_new=np.linspace(0, 3, 1000).reshape(-1, 1)
#     y_proba=log_reg.predict_proba(X_new)
#     plt.plot(X_new, y_proba[:,1], "g--", label="Iris-Virginica")
#     plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not, Iris-Virginica")
#
#     X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
#     y_proba = log_reg.predict_proba(X_new)
#     decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]
#     plt.figure(figsize=(8, 3))
#     plt.plot(X[y==0], y[y==0], "bs")
#     plt.plot(X[y==1], y[y==1], "g^")
#     plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
#     plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
#     plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
#     plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
#     plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
#     plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
#     plt.xlabel("Petal width (cm)", fontsize=14)
#     plt.ylabel("Probability", fontsize=14)
#     plt.legend(loc="center left", fontsize=14)
#     plt.axis([0, 3, -0.02, 1.02])
#     save_fig("logistic_regression_plot")
#     plt.show()
#
#
#     print("line = 579", decision_boundary)
#     print("line = 580", log_reg.predict([[1.7], [1.5]]))
#
#     X = iris["data"][:, (2, 3)]  # petal length, petal width
#     y = (iris["target"] == 2).astype(np.int)
#
#     log_reg = LogisticRegression(C=10 ** 10, random_state=42)
#     log_reg.fit(X, y)
#
#     x0, x1 = np.meshgrid(
#         np.linspace(2.9, 7, 500).reshape(-1, 1),
#         np.linspace(0.8, 2.7, 200).reshape(-1, 1),
#     )
#     X_new = np.c_[x0.ravel(), x1.ravel()]
#
#     y_proba = log_reg.predict_proba(X_new)
#
#     plt.figure(figsize=(10, 4))
#     plt.plot(X[y == 0, 0], X[y == 0, 1], "bs")
#     plt.plot(X[y == 1, 0], X[y == 1, 1], "g^")
#
#     zz = y_proba[:, 1].reshape(x0.shape)
#     contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
#
#     left_right = np.array([2.9, 7])
#     boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
#
#     plt.clabel(contour, inline=1, fontsize=12)
#     plt.plot(left_right, boundary, "k--", linewidth=3)
#     plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
#     plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
#     plt.xlabel("Petal length", fontsize=14)
#     plt.ylabel("Petal width", fontsize=14)
#     plt.axis([2.9, 7, 0.8, 2.7])
#     save_fig("logistic_regression_contour_plot")
#     plt.show()
#
#
#     ##  sotmax
#     X = iris["data"][::, (2, 3)]  # petal length(花瓣长度), petal width
#     y = iris["target"]
#
#     softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
#     softmax_reg.fit(X, y)
#
#     print("line = 627", softmax_reg.predict([[5, 2]]))
#     print("line = 628", softmax_reg.predict_proba([5, 2]))
#
#     x0, x1 = np.meshgrid(
#         np.linspace(0, 8, 500).reshape(-1, 1),
#         np.linspace(0, 3.5, 200).reshape(-1, 1),
#     )
#     X_new = np.c_[x0.ravel(), x1.ravel()]
#
#     y_proba = softmax_reg.predict_proba(X_new)
#     y_predict = softmax_reg.predict(X_new)
#
#     zz1 = y_proba[:, 1].reshape(x0.shape)
#     zz = y_predict.reshape(x0.shape)
#
#     plt.figure(figsize=(10, 4))
#     plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
#     plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
#     plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")
#
#     from matplotlib.colors import ListedColormap
#
#     custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
#
#     plt.contourf(x0, x1, zz, cmap=custom_cmap)
#     contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
#     plt.clabel(contour, inline=1, fontsize=12)
#     plt.xlabel("Petal length", fontsize=14)
#     plt.ylabel("Petal width", fontsize=14)
#     plt.legend(loc="center left", fontsize=14)
#     plt.title('Figure 4-25. Softmax Regression decision boundaries')
#     plt.axis([0, 7, 0, 3.5])
#     save_fig("softmax_regression_contour_plot")
#     plt.show()