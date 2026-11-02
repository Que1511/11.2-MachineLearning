"""
基于物品的协同过滤
    compute_similarity()   计算电影之间的相似度矩阵        —— 公式 (7.1)(7.2) 用在电影向量上
    predict_rating()       预测某个用户对某部电影的评分    —— 公式 (7.7′)，减去电影评分均值
"""
import numpy as np


def compute_similarity(rating_matrix):
    """
    【功能】计算所有电影两两之间的相似度。
           每部电影的评分向量是评分矩阵的一列。
           两部电影的相似度：先求两部电影的欧式距离 d（只比较两部电影都被评过分的用户），再转换为相似度 1 / (1 + d)。
           没有共同评分的用户时，相似度为 0；电影与自己的相似度（对角线）也为 0。

    【输入】rating_matrix : 评分矩阵，numpy 数组，形状 (用户数, 电影数)。
                           rating_matrix[user_index, movie_index] 是该用户对该电影的评分（1~5），0 表示没评过。

    【输出】similarity_matrix : 电影相似度矩阵，numpy 数组，形状 (电影数, 电影数)。
                               similarity_matrix[movie_a, movie_b] 是电影 a 与电影 b 的相似度，矩阵对称。
    """
    movie_vectors = rating_matrix.T                                # 每部电影的评分向量作为一行：(电影数, 用户数)
    num_movies = movie_vectors.shape[0]
    similarity_matrix = np.zeros((num_movies, num_movies))
    rated = movie_vectors > 0                                      # 每部电影被哪些用户评过
    for movie_a in range(num_movies):
        common = rated[movie_a] & rated                            # 两部电影都被评过分的用户
        diff = movie_vectors[movie_a] - movie_vectors
        distance_sq = np.where(common, diff ** 2, 0).sum(axis=1)   # 只累加共同评分用户上的平方差
        distance = np.sqrt(distance_sq)
        sim = np.where(common.any(axis=1), 1.0 / (1.0 + distance), 0.0)
        sim[movie_a] = 0.0                                         # 对角线为 0
        similarity_matrix[movie_a] = sim
    return similarity_matrix


def predict_rating(rating_matrix, similarity_matrix, user_index, movie_index, k_neighbors):
    """
    【功能】预测用户对一部电影的评分（减去电影评分均值的改进公式）。
           近邻：在该用户评过分的其他电影中，与这部电影相似度最高的 k_neighbors 部电影。
           电影的评分均值：这部电影所有评分的平均值（只统计评过分的用户，0 不算）。
           预测分 = 这部电影的评分均值
                    + Σ(近邻电影与这部电影的相似度 × (该用户对近邻电影的评分 − 近邻电影的评分均值)) ÷ Σ(近邻电影与这部电影的相似度)
                                                                                        —— 公式 (7.7′)
           相似度之和为 0 时，预测分为这部电影的评分均值；预测分超出 1~5 时，截断到 1~5。

    【输入】rating_matrix     : 评分矩阵，形状 (用户数, 电影数)，0 表示没评过
           similarity_matrix : compute_similarity() 返回的电影相似度矩阵，形状 (电影数, 电影数)
           user_index        : 要预测的用户在矩阵中的行下标（从 0 开始）
           movie_index       : 要预测的电影在矩阵中的列下标（从 0 开始）
           k_neighbors       : 近邻个数 K（int）

    【输出】predicted_rating : 预测评分（float）
    """
    num_movies = rating_matrix.shape[1]
    # 每部电影的评分均值：只统计评过分的用户（0 不算）
    rating_count = (rating_matrix > 0).sum(axis=0)
    movie_mean = rating_matrix.sum(axis=0) / np.maximum(rating_count, 1)
    # 近邻候选：该用户评过的其他电影（目标电影不算）
    other_movies = np.arange(num_movies) != movie_index
    candidates = np.where((rating_matrix[user_index] > 0) & other_movies)[0]
    if len(candidates) == 0:
        return float(movie_mean[movie_index])
    similarities = similarity_matrix[movie_index, candidates]
    k_neighbors = min(k_neighbors, len(candidates))
    neighbors = candidates[np.argsort(-similarities)[:k_neighbors]]   # 相似度最高的 K 部近邻电影
    weights = similarity_matrix[movie_index, neighbors]
    weight_sum = weights.sum()
    if weight_sum == 0:                                               # 相似度之和为 0，预测分为目标电影均值
        return float(movie_mean[movie_index])
    deviation = rating_matrix[user_index, neighbors] - movie_mean[neighbors]
    predicted = movie_mean[movie_index] + (weights * deviation).sum() / weight_sum
    return float(np.clip(predicted, 1.0, 5.0))                        # 公式 (7.7′)，截断到 1~5
