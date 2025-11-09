# 开发时间：2025/6/20 21:25
import numpy as np
from itertools import combinations

def adjacency_matrix(sequence):
    """计算邻接矩阵"""
    n = len(sequence)
    A = np.zeros((n,n))
    for i in range (n):
        for j in range(n):
            A[i,j] = abs(sequence[i]-sequence[j])
    return A

def ratio_matrix(sequence):
    """计算比例矩阵"""
    k = len(sequence)
    pairs = list(combinations(range(k),2)) # C(k,2) 个 Δp_ij
    num_dist = len(pairs)

    distances = np.array([abs(sequence[i] - sequence[j]) for i,j in pairs ])
    R = np.zeros((num_dist, num_dist))

    for m in range (num_dist):
        for n in range (num_dist):
            if distances[n] != 0:
                R[m,n] = distances[m] / distances[n]
            else:
                R[m,n] = 0.0
    return R

def pattern_distance(X1, A1, R1, X2, A2, R2):
    """
    输入：两组特征点的模式三元组（X,A, R）
    输出：序列模式距离SPD      1 / (1 + exp(-x))
    """
    pos_dist = 1 / (1 + np.exp(-np.linalg.norm(np.array(X1) - np.array(X2))))
    adj_dist = 1 / (1 + np.exp(-np.linalg.norm(A1 - A2,ord='fro')))
    ratio_dist = np.linalg.norm(R1 - R2,ord='fro')
    SPD = pos_dist + adj_dist + ratio_dist
    return SPD

def match_index(seq1, seq2):
    """
    输入：两组特征点位置序列
    输出：PMI值，越大相似度越高
    """
    X1 = np.array(seq1)
    X2 = np.array(seq2)
    A1 = adjacency_matrix(X1)
    A2 = adjacency_matrix(X2)
    R1 = ratio_matrix(X1)
    R2 = ratio_matrix(X2)
    SPD = pattern_distance(X1, A1, R1, X2, A2, R2)
    PMI = 1.0/(SPD)
    return PMI


"""# 图像特征点序列
image_seq = [10, 20, 40]

# 加速度特征点序列
acc_seq = [11, 22, 40]

# 计算 PMI
pmi = match_index(image_seq, acc_seq)
print("Pattern Matching Index (PMI):", pmi)"""