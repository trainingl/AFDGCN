import numpy as np
import pandas as pd

# 1.考虑节点的连通性和距离关系
def get_adjacency_matrix(distance_df_filename, num_of_vertices, type='connectivity', id_filename=None):
    """
    :param distance_df_filename: str, csv边信息文件路径
    :param num_of_vertices: int, 节点数量
    :param type: str, {"connectivity", "distance"}
    :param id_filename: str, 节点信息文件
    :return:
    """
    num_sensors = num_of_vertices
    A = np.zeros((num_sensors, num_sensors), dtype=np.float32)  # N * N
    if id_filename:
        with open(id_filename, 'r') as f:
            # 建立索引表，注意分割符
            sensor_ids = f.read().strip().split('\n')
            id_dict = {
                int(i): idx for idx, i in enumerate(sensor_ids)
            }
            df = pd.read_csv(distance_df_filename)
            for row in df.values:
                if row[0] not in id_dict or row[1] not in id_dict:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                # 无向图还是有向图
                A[id_dict[i], id_dict[j]] = 1 / distance
                A[id_dict[j], id_dict[i]] = 1 / distance
            return A

    df = pd.read_csv(distance_df_filename)
    for row in df.values:
        if len(row) != 3:
            continue
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        if type == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type == 'distance':
            A[i, j] = 1 / distance
            A[i, j] = 1 / distance
        else:
            raise ValueError("type error, must be connectivity or distance!")
    return A

# 2.高斯核函数构建邻接矩阵
def get_Gaussian_matrix(distance_df_filename, num_of_vertices, normalized_k=0.1, id_filename=None):
    num_sensors = num_of_vertices
    A = np.zeros((num_sensors, num_sensors), dtype=np.float32)  # N * N
    A[:] = np.inf   # 初始化无穷大（表示无穷远）

    if id_filename:
        with open(id_filename, 'r') as f:
            # 建立索引表，注意分割符
            sensor_ids = f.read().strip().split('\n')
            id_dict = {
                int(i): idx for idx, i in enumerate(sensor_ids)
            }
            df = pd.read_csv(distance_df_filename)
            for row in df.values:
                if row[0] not in id_dict or row[1] not in id_dict:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                # 无向图还是有向图
                A[id_dict[i], id_dict[j]] = distance
                A[id_dict[j], id_dict[i]] = distance
    else:
        df = pd.read_csv(distance_df_filename)
        for row in df.values:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            # 区分数据是有向图还是无向图
            A[i, j] = distance
            A[j, i] = distance
    # 使用高斯核处理
    distances = A[~np.isinf(A)].flatten()
    std = distances.std()   # 计算距离的方差
    adj_mx = np.exp(-np.square(A/std))
    adj_mx[adj_mx < normalized_k] = 0
    # 返回处理好的邻接矩阵
    return adj_mx


if __name__ == '__main__':
    adj = get_adjacency_matrix("../data/PEMS08/PEMS08.csv", 170, type='distance', id_filename=None)
    print(adj.shape)
    adj = get_Gaussian_matrix("../data/PEMS03/PEMS03.csv", 358, normalized_k=0.1, id_filename="../data/PEMS03/PEMS03.txt")
    print(adj.shape)