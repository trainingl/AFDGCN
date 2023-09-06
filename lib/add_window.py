import numpy as np

def Add_Window_Horizon(data, window=12, horizon=1, single=False):
    '''滑动窗口采样
    :param data: shape [B, N, D]
    :param window:
    :param horizon:
    :return: X is [B, W, N, D], Y is [B, H, N, D]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      # windows
    Y = []      # horizon
    index = 0   # 起始索引
    if single:
        # 单步预测
        while index < end_index:
            X.append(data[index : index+window])
            Y.append(data[index + window+horizon - 1 : index + window + horizon])
            index = index + 1
    else:
        # 多步预测
        while index < end_index:
            X.append(data[index : index+window])
            Y.append(data[index + window : index + window + horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

