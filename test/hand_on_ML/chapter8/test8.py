
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)


from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from six.moves import urllib



# To plot pretty figures
##%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR =  os.path.dirname(os.path.realpath(__file__))
CHAPTER_ID = "08_ Dimensionality Reduction -- unsupervised_learning"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "../images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension=".png"):
    ##path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


if __name__ == '__main__':
    np.random.seed(4)
    m = 60
    w1, w2 = 0.1, 0.3
    noise = 0.1
    # 生成噪音参数
    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    X = np.empty((m, 3))  ## 原始数据
    X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2  ##  第一列
    X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2 ##  第二列
    X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)  ##  第三列
    ##  获取与均值的偏差
    X_centered = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_centered)  ## 对均值偏差 做svd矩阵分解
    c1 = Vt.T[:, 0]  ##  第一条主成分  Vt.T   .T  转置
    c2 = Vt.T[:, 1]  ##  第二条主成分  vt第二行

    m, n = X.shape

    S = np.zeros(X_centered.shape)
    S[:n, :n] = np.diag(s)  ##  diag  输入array 是一维时， 输出是一个以一维数组为对角线的矩阵
                            ##  输入为二维矩阵时， 输出矩阵的对角线元素

    np.allclose(X_centered, U.dot(S).dot(Vt))  ## 比较两个输入是否相同

    W2 = Vt.T[:, :2]
    X2D = X_centered.dot(W2) ##  投影到二维空间

    X2D_using_svd = X2D
    ##  使用pca类 降维到二维
    pca = PCA(n_components=2)
    X2D = pca.fit_transform(X)


    print("line = 126 X2D[:5] = {}".format(X2D[:5]))
    print("line = 127 X2D_using_svd[:5] = {}".format(X2D_using_svd[:5]))
    print("line = 128 np.allclose(X2D, -X2D_using_svd) = {}".format(np.allclose(X2D, -X2D_using_svd)))
    ##  重建3D数据
    X3D_inv = pca.inverse_transform(X2D)

    print("line = 132 np.allclose(X3D_inv, X) = ".format(np.allclose(X3D_inv, X)))

    ## 计算重建误差
    print("line = 135 np.mean(np.sum(np.square(X3D_inv - X), axis=1)) = {}".format( np.mean(np.sum(np.square(X3D_inv - X), axis=1))))
    ## SVD中矩阵逆变换
    X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])
    ##
    print("line = 139 np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean) = {}".format(np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_)))
    print("line = 140 pca.components_ = {}".format(pca.components_))
    print("line = 141 Vt[:2] = {}".format(Vt[:2]))

    ##  解释方差比
    print("line = 144 pca.explained_variance_ratio_".format(pca.explained_variance_ratio_))
    ##  方差损失率
    print("line = 146 ", 1 - pca.explained_variance_ratio_.sum())
    ##  使用svd方法后计算解释的方差比
    print("line = 148", np.square(s) / np.square(s).sum())

    axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

    x1s = np.linspace(axes[0], axes[1], 10)
    x2s = np.linspace(axes[2], axes[3], 10)
    x1, x2 = np.meshgrid(x1s, x2s)

    C = pca.components_  ##  pca 主成分
    R = C.T.dot(C)   ##
    z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

    fig = plt.figure(figsize=(6, 3.8))
    ax = fig.add_subplot(111, projection='3d')

    X3D_above = X[X[:, 2] > X3D_inv[:, 2]]  ##  设置上层点
    X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]  ##  设置下层点
    ##  画出下层点
    ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)
    ##  画出超平面
    ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
    np.linalg.norm(C, axis=0)
    ax.add_artist(
        Arrow3D([0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
    ax.add_artist(
        Arrow3D([0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
    ax.plot([0], [0], [0], "k.")

    for i in range(m):
        if X[i, 2] > X3D_inv[i, 2]:
            ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
        else:
            ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#505050")

    ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
    ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
    ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    save_fig("dataset_3d_plot")
    plt.show()
    ##  画出映射后的  二维图像
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ax.plot(X2D[:, 0], X2D[:, 1], "k+")
    ax.plot(X2D[:, 0], X2D[:, 1], "k.")
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel("$z_1$", fontsize=18)
    ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
    ax.axis([-1.5, 1.3, -1.2, 1.2])
    ax.grid(True)
    save_fig("dataset_2d_plot")
    ##  生成瑞士卷
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

    axes = [-11.5, 14, -2, 23, -12, 15]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    save_fig("swiss_roll_plot")
    plt.show()

    plt.figure(figsize=(11, 4))

    plt.subplot(121)  ##  瑞士卷 映射一
    plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
    plt.axis(axes[:4])
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)

    plt.subplot(122) ##  瑞士卷 映射二
    plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.grid(True)

    save_fig("squished_swiss_roll_plot")
    plt.show()
## 瑞士卷 不同的映射方式 生成不同的二维图像
    axes = [-11.5, 14, -2, 23, -12, 15]

    x2s = np.linspace(axes[2], axes[3], 10)
    x3s = np.linspace(axes[4], axes[5], 10)
    x2, x3 = np.meshgrid(x2s, x3s)
    ##  瑞士卷 正负值 沿x2,x3形成的平面切分
    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, projection='3d')

    positive_class = X[:, 0] > 5 ## 设置正负阈值
    X_pos = X[positive_class]  ## 获取正例掩码
    X_neg = X[~positive_class]  ##  获取负例掩码
    ax.view_init(10, -70)
    ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^") ##  负例显示黄色三角
    ax.plot_wireframe(5, x2, x3, alpha=0.5)
    ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs") ##  正例显示绿色方块
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    save_fig("manifold_decision_boundary_plot1")
    plt.show()
    ##  正负值 沿x2,x3轴形成的平面切分后 映射成的二维图像
    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)

    plt.plot(t[positive_class], X[positive_class, 1], "gs")
    plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

    save_fig("manifold_decision_boundary_plot2")
    plt.show()
    ##  瑞士卷 沿x1,x3 形成的平面 切分正负值
    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, projection='3d')

    positive_class = 2 * (t[:] - 4) > X[:, 1]
    X_pos = X[positive_class]
    X_neg = X[~positive_class]
    ax.view_init(10, -70)
    ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
    ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    save_fig("manifold_decision_boundary_plot3")
    plt.show()

    ## 正负值沿x1, x3轴形成的平面映射后的图像
    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)

    plt.plot(t[positive_class], X[positive_class, 1], "gs")
    plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
    plt.plot([4, 15], [0, 22], "b-", linewidth=2)
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

    save_fig("manifold_decision_boundary_plot4")
    plt.show()

#
#
### PCA
    angle = np.pi / 5
    stretch = 5
    m = 200
    ##  生成原始数据
    np.random.seed(3) ##并对其进行加强和选择
    X = np.random.randn(m, 2) / 10
    X = X.dot(np.array([[stretch, 0], [0, 1]]))  # stretch
    X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])  # rotate
    ##  生成三种不同方向向量
    u1 = np.array([np.cos(angle), np.sin(angle)])
    u2 = np.array([np.cos(angle - 2 * np.pi / 6), np.sin(angle - 2 * np.pi / 6)])
    u3 = np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])
    ##  对三种不同向量 生成映射点
    X_proj1 = X.dot(u1.reshape(-1, 1))
    X_proj2 = X.dot(u2.reshape(-1, 1))
    X_proj3 = X.dot(u3.reshape(-1, 1))

    plt.figure(figsize=(8, 4))
    plt.subplot2grid((3, 2), (0, 0), rowspan=3) ## 左边的原始点和映射向量
    plt.plot([-1.4, 1.4], [-1.4 * u1[1] / u1[0], 1.4 * u1[1] / u1[0]], "k-", linewidth=1)
    plt.plot([-1.4, 1.4], [-1.4 * u2[1] / u2[0], 1.4 * u2[1] / u2[0]], "k--", linewidth=1)
    plt.plot([-1.4, 1.4], [-1.4 * u3[1] / u3[0], 1.4 * u3[1] / u3[0]], "k:", linewidth=2)
    plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)
    plt.axis([-1.4, 1.4, -1.4, 1.4])
    plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k',
              ec='k')
    plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k',
              ec='k')
    plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
    plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)
    ##  映射到u1的点分布情况
    plt.subplot2grid((3, 2), (0, 1))
    plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
    plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])
    plt.grid(True)
    ##  映射到u2的点分布情况
    plt.subplot2grid((3, 2), (1, 1))
    plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
    plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])
    plt.grid(True)
    ##  映射到u3的点分布
    plt.subplot2grid((3, 2), (2, 1))
    plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
    plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.axis([-2, 2, -1, 1])
    plt.xlabel("$z_1$", fontsize=18)
    plt.grid(True)

    save_fig("pca_best_projection")
    plt.show()
    ##  使用pca 方法 设置方差解释率大于95%时的维度
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    ##  直接设置 n_components 相应的方差解释率
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)


    mnist = fetch_mldata('MNIST original', data_home='/home/inzahgi/test/jupyter/hand_on_ML/Hands-on-Machine-Learning/datasets')


    X = mnist["data"]
    y = mnist["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    ##  获取mnist图像 方差解释率为95%时的维度
    pca = PCA()
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1

    print("line = 405 d = {}".format(d))

    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_train)

    print("line = 410 pca.n_components_ = {}", pca.n_components_)
    print("line = 411 np.sum(pca.explained_variance_ratio_) = {}".format(np.sum(pca.explained_variance_ratio_)))
    ##  设置pca为154维度后 生成 映射数据和逆转数据
    pca = PCA(n_components=154)
    X_reduced = pca.fit_transform(X_train)
    X_recovered = pca.inverse_transform(X_reduced)
    ##  画出原始点和压缩后的点
    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plot_digits(X_train[::2100])
    plt.title("Original", fontsize=16)
    plt.subplot(122)
    plot_digits(X_recovered[::2100])
    plt.title("Compressed", fontsize=16)

    save_fig("mnist_compression_plot")

    X_reduced_pca = X_reduced

##  incremental PCA  增量pca
    n_batches = 100
    inc_pca = IncrementalPCA(n_components=154)
    ##  分小批次 训练
    for X_batch in np.array_split(X_train, n_batches):
        inc_pca.partial_fit(X_batch)

    X_reduced = inc_pca.transform(X_train)

    X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced)
    ##  对比 原始数据和 增量pca降维后还原的数据
    plt.figure(figsize=(7, 4))
    plt.subplot(121)
    plot_digits(X_train[::2100])
    plt.subplot(122)
    plot_digits(X_recovered_inc_pca[::2100])
    plt.tight_layout()

    X_reduced_inc_pca = X_reduced

    print("line = 449 np.allclose(pca.mean_, inc_pca.mean_) = {}".format(np.allclose(pca.mean_, inc_pca.mean_)))
    print("line = 450 np.allclose(X_reduced_pca, X_reduced_inc_pca) = {}".format(np.allclose(X_reduced_pca, X_reduced_inc_pca)))

    filename = "my_mnist.data"
    m, n = X_train.shape
    ## 通过内存映射的方式获取 数据
    X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
    X_mm[:] = X_train
    ## 现在删除memmap（）对象将触发其Python终结器，确保将数据保存到磁盘。
    del X_mm

    X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))

    batch_size = m // n_batches
    inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
    inc_pca.fit(X_mm)
    ##  随机pca
    rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
    X_reduced = rnd_pca.fit_transform(X_train)


    ##  Kernel PCA
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
    ##  生成rbf_pca 类
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
    X_reduced = rbf_pca.fit_transform(X)  ## 利用rbf核函数 对瑞士卷降维
    ##  线性核  rbf核  sigmoid冲击函数核
    lin_pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
    sig_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

    y = t > 6.9
    ## 分别输出 线性 rbf sigmoid 核函数降维后的图像
    plt.figure(figsize=(11, 4))
    for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
                                (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
        X_reduced = pca.fit_transform(X)
        if subplot == 132:
            X_reduced_rbf = X_reduced

        plt.subplot(subplot)
        # plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
        # plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
        plt.title(title, fontsize=14)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
        plt.xlabel("$z_1$", fontsize=18)
        if subplot == 131:
            plt.ylabel("$z_2$", fontsize=18, rotation=0)
        plt.grid(True)

    save_fig("kernel_pca_plot")
    plt.show()
    ##  绘制重建 rbf 降维后的图像
    plt.figure(figsize=(6, 5))
    ##  重建原始数据
    X_inverse = rbf_pca.inverse_transform(X_reduced_rbf)

    ax = plt.subplot(111, projection='3d')
    ax.view_init(10, -70)
    ax.scatter(X_inverse[:, 0], X_inverse[:, 1], X_inverse[:, 2], c=t, cmap=plt.cm.hot, marker="x")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    save_fig("preimage_plot", tight_layout=False)
    plt.show()
    ##  画出降维后的映射图像
    X_reduced = rbf_pca.fit_transform(X)

    plt.figure(figsize=(11, 4))
    plt.subplot(132)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot, marker="x")
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
    ##  对kpca算法 使用网格搜索 找出最佳维度
    clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

    param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X, y)
    ##  打印网格搜索到的最佳参数
    print("line = 541 grid_search.best_params_ = {}".format(grid_search.best_params_))
    ##  设置 fit_inverse_transform 为true  使用监督训练模型的方式  以降维数据为输入 原始数据为 目标数据  调整参数
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433,
                        fit_inverse_transform=True)

    X_reduced = rbf_pca.fit_transform(X)
    X_preimage = rbf_pca.inverse_transform(X_reduced)

    print("line = 549  mean_squared_error(X, X_preimage) = {}".format(mean_squared_error(X, X_preimage)))



##  LLE  流型嵌入技术

    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
    X_reduced = lle.fit_transform(x)

    plt.title("Unrolled swiss roll using LLE", fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18)
    plt.axis([-0.065, 0.055, -0.1, 0.12])
    plt.grid(True)

    save_fig("lle_unrolling_plot")
    plt.show()


##  其他降维技术
    ##  多维缩放MD  )在尝试保留实例之间的距离时降低了维度
    mds = MDS(n_components=2, random_state=42)
    X_reduced_mds = mds.fit_transform(X)
    ##  isomap 通过将每个实例连接到最近的邻居来创建图形，然后在尝试保持实例之间的测地距离时降低维数
    isomap = Isomap(n_components=2)
    X_reduced_isomap = isomap.fit_transform(X)
    ##t分布随机邻域嵌入- t-Distributed Stochastic Neighbor Embedding (t-SNE) 降低了维数同时试图让类似的实例保持接近，
    # 并将不同的实例分开。它主要用于可视化，特别地，可视化高维空间中的实例簇
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced_tsne = tsne.fit_transform(X)
    ## 线性判别分析- Linear Discriminant Analysis（LDA）实际上是一种分类算法，但是在训练期间，它会学习类之间最具辨别力的轴
    # 然后可以使用这些轴来定义投影数据的超平面。好处是投影将使类与类尽可能远离，所以在运行另一种分类算法（如SVM分类器）之前
    # ，LDA是一种降低维数的好方法
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_mnist = mnist["data"]
    y_mnist = mnist["target"]
    lda.fit(X_mnist, y_mnist)
    X_reduced_lda = lda.transform(X_mnist)

    titles = ["MDS", "Isomap", "t-SNE"]
    ##  画出 mds  isomap  t-sne 三种降维后的图像
    plt.figure(figsize=(11, 4))
    for subplot, title, X_reduced in zip((131, 132, 133), titles,
                                         (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
        plt.subplot(subplot)
        plt.title(title, fontsize=14)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
        plt.xlabel("$z_1$", fontsize=18)
        if subplot == 131:
            plt.ylabel("$z_2$", fontsize=18, rotation=0)
        plt.grid(True)

    save_fig("other_dim_reduction_plot")
    plt.show()



