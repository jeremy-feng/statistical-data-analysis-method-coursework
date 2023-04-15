# 稀疏高维协方差矩阵的 Thresholding 估计方法

高维协方差矩阵的一个重要特征就是许多维度之间的协方差非常接近于 0，一个自然的想法就是将矩阵中绝对值太小的元素设为 0，这种方法就是 Thresholding（门限）：通过设定某个门限，将绝对值小于该门限的元素设为 0，只保留绝对值大于或等于该门限的元素。

通过 Thresholding 估计方法，我们可以得到一个比样本协方差矩阵更稀疏的估计。学术界提出了两种设置 Thresholding 的方法：Universal thresholding（统一截断）和 Adaptive thresholding（自适应截断）。前者对矩阵中的每一个元素均采用相同的门限标准，而后者基于样本协方差估计的标准误自适应地为每个元素设定门槛。

本文使用模拟的高斯分布数据和真实的高维 DNA 基因数据，比较了 Universal thresholding 和 Adaptive thresholding 的估计效果，所得结果与 Tony Cai & Weidong Liu (2011) 中的结果基本一致。

![png](README-image/output_25_0.png)

## 导入包

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from rich.progress import Progress
# 设定随机种子
np.random.seed(42)
```

## 定义类和函数，用于生成协方差矩阵、实施各种截断方法

```python
class SigmaGenerator:
    """
    生成 Model 1 的协方差矩阵
    """

    def __init__(self, p):
        # 必须保证 p 为偶数，否则无法生成 Model 1 的协方差矩阵
        assert p % 2 == 0, "p must be even"
        self.p = int(p / 2)

    def A_1_sigma(self):
        """
        生成 A1
        """
        # 初始化 A1
        matrix = np.zeros((self.p, self.p))
        # 生成 A1 中的第 i 行 第 j 列的值
        for i in range(self.p):
            for j in range(self.p):
                matrix[i][j] = max(0, 1 - abs(i - j) / 10)
        return matrix

    def A_2_sigma(self):
        """
        生成 A2
        """
        # 初始化 A2，对角线元素均为 4，其他元素均为 0
        matrix = 4 * np.eye(self.p)
        return matrix

    def gen_sigma(self):
        """
        生成 Model 1 的协方差矩阵
        """
        # 用 A1 和 A2 两个 bolck 组成 sigma
        matrix = np.block(
            [
                [self.A_1_sigma(), np.zeros((self.p, self.p))],
                [np.zeros((self.p, self.p)), self.A_2_sigma()],
            ]
        )
        return matrix

class Thresholding:
    def __init__(self, vectors, labels=None):
        self.vectors = vectors
        self.labels = labels
        self.n = vectors.shape[0]
        self.p = vectors.shape[1]

    def cal_cov(self, vectors=None):
        """
        计算协方差矩阵

        Args:
            vectors: 多个向量

        Returns:
            cov: 协方差矩阵
        """
        # 如果没有传入 vectors，则使用 self.vectors
        if vectors is None:
            vectors = self.vectors
        # 如果传入 vectors，则使用传入的 vectors
        cov = np.cov(vectors, rowvar=False)
        return cov

    def cal_risk(
        self, cov_1: np.ndarray, cov_2: np.ndarray, norm_type: str = "operator"
    ):
        """
        计算两个协方差矩阵之差的 norm

        Args:
            cov_1: 协方差矩阵 1
            cov_2: 协方差矩阵 2
            norm_type: norm 的类型，可选 'Operator norm', 'Matrix l1 norm' or 'Frobenius norm'

        Returns:
            risk: 两个协方差矩阵之差的 norm
        """
        # 断言 cov_1 和 cov_2 的形状相同
        assert cov_1.shape == cov_2.shape, "cov_1 and cov_2 must have the same shape"
        diff = cov_1 - cov_2
        if norm_type == "Operator norm":
            risk = np.linalg.norm(diff, ord=2)
        elif norm_type == "Matrix l1 norm":
            risk = np.linalg.norm(diff, ord=1)
        elif norm_type == "Frobenius norm":
            risk = np.linalg.norm(diff, ord="fro")
        else:
            raise ValueError(
                "norm_type must be 'Operator norm', 'Matrix l1 norm' or 'Frobenius norm'"
            )
        return risk

    def thresholding(self, cov: np.ndarray, threshold: np.ndarray):
        """
        对协方差矩阵进行 thresholding

        Args:
            cov: 协方差矩阵
            threshold: 截断标准矩阵

        Returns:
            cov: 截断后的协方差矩阵
        """
        # 断言 cov 和 threshold 的形状相同
        assert (
            cov.shape == threshold.shape
        ), "cov and threshold must have the same shape"
        # copy() 是为了防止修改原矩阵
        cov = cov.copy()
        # 对协方差矩阵进行 thresholding
        # 将绝对值小于 threshold 的元素置为 0
        cov[np.abs(cov) < threshold] = 0
        return cov

    def cal_theta_hat(self):
        """
        计算 theta_hat

        Returns:
            theta_hat: theta_hat
        """
        # 计算 sigma_hat
        sigma_hat = self.cal_cov()
        # 计算 theta_hat
        theta_hat = np.zeros((self.p, self.p))
        for i in range(self.p):
            x_i_bar = np.mean(self.vectors[:, i])
            for j in range(self.p):
                x_j_bar = np.mean(self.vectors[:, j])
                for k in range(self.n):
                    theta_hat[i][j] += (
                        (self.vectors[k][i] - x_i_bar) * (self.vectors[k][j] - x_j_bar)
                        - sigma_hat[i][j]
                    ) ** 2
        theta_hat /= self.n
        return theta_hat

    def get_best_threshold_using_cv(
        self,
        method: str = "universal",
        norm_type: str = "operator",
        N: int = 10,
        split_method: str = "KFold",
    ):
        """
        使用交叉验证对样本进行 thresholding，选取最优的截断标准矩阵 threshold

        Args:
            method: thresholding 的方法，可选 universal, adaptive, adaptive2
            norm_type: norm 的类型，可选 operator, l1, frobenius
            N: 交叉验证的折数
            split_method: 交叉验证的方法，可选 KFold, StratifiedKFold

        Returns:
            threshold: 最优的截断标准矩阵

        """
        # 计算样本协方差矩阵中的最大元素值
        max_cov = np.max(np.abs(self.cal_cov()))
        # 构造 delta 的候选集，最小为 0，最大为原协方差矩阵中的最大值，共 5 个候选值
        delta_candidates = np.linspace(0, max_cov, 5)
        if method == "Universal":
            # 构造 threshold 的候选集，它是一个 1 维数组，每个元素代表一个 threshold
            threshold_candidates = delta_candidates * np.sqrt(np.log(self.p) / self.n)
            # 将 threshold_candidates 转换为列表，列表中的元素为 self.p * self.p 的矩阵，矩阵中的元素均为 threshold
            threshold_candidates = [
                np.full((self.p, self.p), threshold)
                for threshold in threshold_candidates
            ]
        elif method == "Adaptive":
            # 计算 theta_hat
            theta_hat = self.cal_theta_hat()
            # 构造 threshold 的候选集
            threshold_candidates = [
                delta * np.sqrt(theta_hat * np.log(self.p) / self.n)
                for delta in delta_candidates
            ]
        elif method == "Adaptive2":
            # 计算 theta_hat
            theta_hat = self.cal_theta_hat()
            # 使用 delta = 2
            return 2 * np.sqrt(theta_hat * np.log(self.p) / self.n)
        else:
            raise ValueError("method must be Universal, Adaptive or Adaptive2")
        # 构造交叉验证划分样本的实例
        if split_method == "KFold":
            spliter = KFold(n_splits=N, shuffle=True, random_state=42)
        elif split_method == "StratifiedKFold":
            spliter = StratifiedKFold(n_splits=N, shuffle=True, random_state=42)
        else:
            raise ValueError("split_method must be KFold or StratifiedKFold")
        # 构造 risks 列表，用于存储每次交叉验证的 risk
        risks = []
        # 构造交叉验证的分割索引
        if split_method == "KFold":
            split_index = spliter.split(self.vectors)
        elif split_method == "StratifiedKFold":
            split_index = spliter.split(self.vectors, self.labels)
        else:
            raise ValueError("split_method must be KFold or StratifiedKFold")
        # 遍历 threshold_candidates 中的每个 threshold
        for threshold in threshold_candidates:
            risk = 0
            for n_1_index, n_2_index in split_index:
                # 计算训练集和测试集的协方差矩阵
                cov_n_1 = self.cal_cov(self.vectors[n_1_index])
                cov_n_2 = self.cal_cov(self.vectors[n_2_index])
                # 将 cov_n_1 进行 thresholding
                cov_n_1 = self.thresholding(cov_n_1, threshold)
                # 计算训练集和测试集的 risk
                risk += self.cal_risk(cov_n_1, cov_n_2, norm_type)
                # 将 risk 的均值添加到 risks 中
            risks.append(risk / N)
        # 选取 risks 中的最小 risk 对应的 threshold
        threshold = threshold_candidates[np.argmin(risks)]
        return threshold
```

## 模拟高斯分布的协方差矩阵估计

### 估计步骤

![image-20230327093230679](README-image/image-20230327093230679.png)

### 构造数据框，存储结果

```python
# 构造双重索引
idx = pd.MultiIndex.from_product(
    [
        ["Operator norm", "Matrix l1 norm", "Frobenius norm"],
        [30, 100, 200],
    ],
    names=["norm_type", "p"],
)
# 定义列名
col_names = ["Universal", "Adaptive", "Adaptive2"]
# 创建空的 DataFrame
table_1 = pd.DataFrame(columns=col_names, index=idx)
```

### 复现 Table 1：计算各维度、各截断方法、各参数下的 risk 均值和标准误

```python
# 重复 100 次，计算 risk
def replication(progress, task, mu, cov, method, norm_type):
    # 定义 risks 列表，用于存储 risk
    risk_list = []
    # 重复 100 次，计算 risk 的均值和标准差
    for _ in range(100):
        # 生成 100 个 p 元正态分布的随机向量
        vectors = np.random.multivariate_normal(mu, cov, 100)
        # 定义 Thresholding 实例
        thresholding = Thresholding(vectors)
        # 计算样本协方差矩阵
        cov_hat = thresholding.cal_cov()
        # 进行 10 次交叉验证，取最优截断标准矩阵
        best_threshold = thresholding.get_best_threshold_using_cv(method, norm_type, N=10)
        # 对样本协方差矩阵进行 thresholding
        cov_hat_after_thresholding = thresholding.thresholding(cov_hat, best_threshold)
        # 计算 risk
        risk = thresholding.cal_risk(cov_hat_after_thresholding, cov, norm_type)
        # 将 risk 加入 risk_list
        risk_list.append(risk)
        # 更新进度条
        progress.update(task, advance=1)
    return risk_list
```

```python
with Progress() as progress:
    task = progress.add_task("[green] 正在计算。..", total=3 * 3 * 3 * 100)
    for p in [30, 100, 200]:
        # 生成真实的 p 元正态分布的均值向量和协方差矩阵
        mu = np.zeros(p)
        cov = SigmaGenerator(p).gen_sigma()
        # 遍历三种截断方法
        for method in ["Universal", "Adaptive", "Adaptive2"]:
            # 遍历三种计算 risk 的方法
            for norm_type in ["Operator norm", "Matrix l1 norm", "Frobenius norm"]:
                # 重复 100 次，计算 risk
                risk_list = replication(
                    progress, task, mu, cov, method, norm_type
                )
                # 计算 risk 的均值和标准差
                risk_mean = np.mean(risk_list)
                risk_std_error = np.std(risk_list) / np.sqrt(100)
                table_1.loc[
                    (norm_type, p), method
                ] = f"{risk_mean:.2f} ({risk_std_error:.2f})"
```

|      norm_type |   p |    Universal |     Adaptive |    Adaptive2 |
| -------------: | --: | -----------: | -----------: | -----------: |
|  Operator norm |  30 |  3.49 (0.05) |  2.03 (0.05) |  1.83 (0.05) |
|                | 100 |  7.34 (0.05) |  2.64 (0.05) |  2.97 (0.05) |
|                | 200 | 11.18 (0.06) |  3.00 (0.04) |  3.75 (0.05) |
| Matrix l1 norm |  30 |  8.88 (0.15) |  3.07 (0.10) |  2.63 (0.06) |
|                | 100 | 25.22 (0.27) |  4.68 (0.10) |  4.77 (0.06) |
|                | 200 | 44.74 (0.37) |  5.28 (0.12) |  5.85 (0.10) |
| Frobenius norm |  30 |  7.31 (0.05) |  4.01 (0.06) |  3.25 (0.05) |
|                | 100 | 22.50 (0.07) |  7.91 (0.09) |  7.52 (0.05) |
|                | 200 | 43.48 (0.11) | 11.11 (0.10) | 11.72 (0.06) |

### 原始论文结果

![image-20230327092153398](README-image/image-20230327092153398.png)

## 真实高维 DNA 基因数据的协方差矩阵估计

### 估计步骤

!!! warning "论文中的笔误"

    右下角倒数第 3 行的 $\hat \sigma_{m}$ 代表的是样本标准差，它的平方才是样本方差。

![image-20230327093054070](README-image/image-20230327093054070.png)

![image-20230327093108272](README-image/image-20230327093108272.png)

### 定义计算 F 统计量的函数

```python
def cal_F_statistics(df: pd.DataFrame, gene: str) -> float:
    """
    计算 F 统计量

    Args:
        df: DataFrame, 数据集
        gene: str, 基因名

    Returns:
        F: float, F 统计量
    """
    # n 是样本数
    n = len(df)
    # k 是类别数
    k = len(df["class"].unique())
    # x_bar 是总体均值
    x_bar = df[gene].mean()
    # 计算 F 统计量的分子
    numerator = 0
    for m in df["class"].unique():
        df_m = df[df["class"] == m]
        # x_bar_m 是第 m 类的均值
        x_bar_m = df_m[gene].mean()
        # n_m 是第 m 类的样本数
        n_m = len(df_m)
        numerator += n_m * (x_bar_m - x_bar) ** 2
    numerator /= k - 1
    # 计算 F 统计量的分母
    denominator = 0
    for m in df["class"].unique():
        df_m = df[df["class"] == m]
        # sigma_m_square 是第 m 类的样本方差
        sigma_m_square = df_m[gene].var()
        # n_m 是第 m 类的样本数
        n_m = len(df_m)
        denominator += (n_m - 1) * sigma_m_square
    denominator /= n - k
    # 计算 F 统计量
    F = numerator / denominator
    return F

```

### 读取数据

```python
df = pd.read_csv('./SRBCT$X+class.csv', index_col=0).iloc[:63]
```

### 筛选出 F 统计量最大的 40 个基因和最小的 160 个基因

```python
# 计算每个基因的 F 统计量
F_statistics = {}
for gene in df.columns[:-1]:
    F_statistics[gene] = cal_F_statistics(df, gene)
# 筛选出 F 统计量最大的 40 个基因和最小的 160 个基因
top_40_genes = sorted(
    F_statistics,
    key=F_statistics.get,
    reverse=True,
)[:40]
bottom_160_genes = sorted(
    F_statistics,
    key=F_statistics.get,
    reverse=True,
)[-160:]
# 提取 top_40_genes 和 bottom_160_genes 中的基因
df = df[top_40_genes + bottom_160_genes + ["class"]]
```

经过筛选后，得到 200 维基因数据，最后一列为样本的类别。

![image-20230327092545489](README-image/image-20230327092545489.png)

### 复现 Figure 3：绘制不同截断方法下的热力图

```python
# 创建 Thresholding 实例
thresholding = Thresholding(vectors=df.iloc[:, :-1].values, labels=df["class"].values)
# 计算样本协方差矩阵
cov_hat = thresholding.cal_cov()
```

```python
def plot_heat_map(method):
    # 进行 5 次交叉验证，取最优截断标准矩阵。使用 StratifiedKFold 进行交叉验证，尽量使四种类别的样本在每折中的比例相同
    best_threshold = thresholding.get_best_threshold_using_cv(
        method, "Frobenius norm", N=5, split_method="StratifiedKFold"
    )
    # 对样本协方差矩阵进行 thresholding
    cov_hat_after_thresholding = thresholding.thresholding(cov_hat, best_threshold)
    # 绘制黑白图片
    plt.imshow(cov_hat_after_thresholding > 0, cmap="gray_r")
    zeros_proportion = 1 - np.count_nonzero(cov_hat_after_thresholding) / np.size(
        cov_hat_after_thresholding
    )
    # 将 zeros_proportion 作为图片的标题
    plt.title("{}: {:.2%} zeros".format(method, zeros_proportion))
    # 导出到本地 png 格式的图片
    plt.savefig(
    "./{}.png".format(method),
    format="png",
    facecolor="white",
    bbox_inches="tight",
)

```

```python
plot_heat_map('Adaptive')
```

![png](README-image/output_25_0.png)

```python
plot_heat_map('Universal')
```

![png](README-image/output_26_0.png)

### 原始论文结果

![image-20230327092938595](README-image/image-20230327092938595.png)
