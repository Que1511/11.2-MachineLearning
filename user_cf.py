"""
基于用户的协同过滤
    compute_similarity()   计算用户之间的相似度矩阵        —— 公式 (7.1)(7.2)
    predict_rating()       预测某个用户对某部电影的评分    —— 公式 (7.5)
"""
import numpy as np


def compute_similarity(rating_matrix):
    """
    【功能】计算所有用户两两之间的相似度。
           两个用户的相似度：先求两人的欧式距离 d（只比较两人都评过分的电影），再转换为相似度 1 / (1 + d)。
           两人没有共同评过分的电影时，相似度为 0；用户与自己的相似度（对角线）也为 0。

    【输入】rating_matrix : 评分矩阵，numpy 数组，形状 (用户数, 电影数)。
                           rating_matrix[user_index, movie_index] 是该用户对该电影的评分（1~5），0 表示没评过。

    【输出】similarity_matrix : 用户相似度矩阵，numpy 数组，形状 (用户数, 用户数)。
                               similarity_matrix[user_a, user_b] 是用户 a 与用户 b 的相似度，矩阵对称。
    """
    num_users = rating_matrix.shape[0]
    similarity_matrix = np.zeros((num_users, num_users))
    rated = rating_matrix > 0                                      # 每个用户评过哪些电影
    for user_a in range(num_users):
        common = rated[user_a] & rated                             # 两人都评过分的电影
        diff = rating_matrix[user_a] - rating_matrix               # 用户 a 与每个用户的评分差
        distance_sq = np.where(common, diff ** 2, 0).sum(axis=1)   # 只累加共同评分电影上的平方差
        distance = np.sqrt(distance_sq)
        sim = np.where(common.any(axis=1), 1.0 / (1.0 + distance), 0.0)
        sim[user_a] = 0.0                                          # 对角线为 0
        similarity_matrix[user_a] = sim
    return similarity_matrix


def predict_rating(rating_matrix, similarity_matrix, user_index, movie_index, k_neighbors):
    """
    【功能】预测用户对一部电影的评分。
           近邻：在评过这部电影的其他用户中，与该用户相似度最高的 k_neighbors 个用户。
           预测分 = Σ(近邻与该用户的相似度 × 近邻对这部电影的评分) ÷ Σ(近邻与该用户的相似度)    —— 公式 (7.5)
           相似度之和为 0 时，预测分为 3。

    【输入】rating_matrix     : 评分矩阵，形状 (用户数, 电影数)，0 表示没评过
           similarity_matrix : compute_similarity() 返回的用户相似度矩阵，形状 (用户数, 用户数)
           user_index        : 要预测的用户在矩阵中的行下标（从 0 开始）
           movie_index       : 要预测的电影在矩阵中的列下标（从 0 开始）
           k_neighbors       : 近邻个数 K（int）

    【输出】predicted_rating : 预测评分（float）
    """
    num_users = rating_matrix.shape[0]
    # 近邻候选：评过这部电影的其他用户（用户自己不算）
    other_users = np.arange(num_users) != user_index
    candidates = np.where((rating_matrix[:, movie_index] > 0) & other_users)[0]
    if len(candidates) == 0:
        return 3.0
    similarities = similarity_matrix[user_index, candidates]
    k_neighbors = min(k_neighbors, len(candidates))
    neighbors = candidates[np.argsort(-similarities)[:k_neighbors]]   # 相似度最高的 K 个近邻
    weights = similarity_matrix[user_index, neighbors]
    weight_sum = weights.sum()
    if weight_sum == 0:                                               # 相似度之和为 0，预测分为 3
        return 3.0
    rating_sum = (weights * rating_matrix[neighbors, movie_index]).sum()
    return float(rating_sum / weight_sum)                             # 公式 (7.5)：相似度加权平均
