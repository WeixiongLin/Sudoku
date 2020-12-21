import numpy as np


def expand(rawLabels):
    """扩充 Dataset 的 Labels
    Args:
        rawLabels: 未扩充的 Labels
        expandedLabels: 扩充后的 Labels
    Return:
        expandedLabels: 扩充之后的数据集
    """
    expandedLabels = np.zeros((rawLabels.shape[0], 20))
    N = rawLabels.shape[0]
    for i in range(N):
        expandedLabels[i][rawLabels[i]-1] = 1
    return expandedLabels
