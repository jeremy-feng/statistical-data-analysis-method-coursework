import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("svg")
plt.rcParams["axes.unicode_minus"] = False
from tqdm import tqdm

# 机器学习模型
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

# Conformal Prediction
from nonconformist.cp import IcpClassifier
from nonconformist.nc import (
    InverseProbabilityErrFunc,
    MarginErrFunc,
    NcFactory,
    ClassifierNc,
)
from nonconformist.base import ClassifierAdapter


# 设置百分数的格式
def to_percent(temp, position):
    return "%1.0f" % (100 * temp) + "%"


def cal_coverage_ratio(prediction_set, y_test):
    coverage_ratio = sum(
        1 for i in range(len(y_test)) if y_test[i] in prediction_set[i]
    ) / len(y_test)
    return coverage_ratio


def cal_useful_coverage_ratio(prediction_set, y_test):
    useful_coverage_ratio = sum(
        1
        for i in range(len(y_test))
        if y_test[i] in prediction_set[i] and len(prediction_set[i]) == 1
    ) / len(y_test)
    return useful_coverage_ratio


def cal_coverage_ratio_for_y_equals_1(prediction_set, y_test):
    coverage_ratio_for_y_equals_1 = sum(
        1
        for i in range(len(y_test))
        if y_test[i] in prediction_set[i] and y_test[i] == 1
    ) / sum(1 for i in range(len(y_test)) if y_test[i] == 1)
    return coverage_ratio_for_y_equals_1


def cal_useful_coverage_ratio_for_y_equals_1(prediction_set, y_test):
    useful_coverage_ratio_for_y_equals_1 = sum(
        1
        for i in range(len(y_test))
        if y_test[i] in prediction_set[i]
        and len(prediction_set[i]) == 1
        and y_test[i] == 1
    ) / sum(1 for i in range(len(y_test)) if y_test[i] == 1)
    return useful_coverage_ratio_for_y_equals_1


def cal_undecided_ratio(prediction_set, y_test):
    undecided_ratio = sum(
        1 for i in range(len(y_test)) if len(prediction_set[i]) != 1
    ) / len(y_test)
    return undecided_ratio
