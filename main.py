"""
实验七  基于协同过滤的电影推荐 —— 整体流程
=============================================
运行：python main.py

需要你完成的函数（共 5 个）：
    main.py      choose_best_k()                          选择最优近邻个数 K
    user_cf.py   compute_similarity()、predict_rating()   基于用户的协同过滤
    item_cf.py   compute_similarity()、predict_rating()   基于物品的协同过滤
其余代码已经写好，不需要修改。
其中 recommend_top_n() 是测试用的方法：用你写的算法真正给用户推荐电影，帮助你理解这些函数写出来有什么用。
"""
import os, time
import numpy as np
import pandas as pd
import user_cf
import item_cf

ROOT = os.path.dirname(os.path.abspath(__file__))
ALGORITHMS = {"user": ("基于用户的协同过滤", user_cf),
              "item": ("基于物品的协同过滤", item_cf)}
CANDIDATE_K = {"user": [5, 10, 20, 30, 50, 80, 120, 200, 400],    # 候选的近邻个数 K
          "item": [2, 3, 5, 8, 10, 15, 20, 30, 50]}


def build_matrix(ratings, num_users, num_movies):
    """评分矩阵：rating_matrix[用户下标, 电影下标] = 评分，没评分记 0（编号从 1 开始，下标从 0 开始）"""
    rating_matrix = np.zeros((num_users, num_movies))
    for user_id, movie_id, rating in zip(ratings.user_id, ratings.movie_id, ratings.rating):
        rating_matrix[user_id - 1, movie_id - 1] = rating
    return rating_matrix


def choose_best_k(algorithm, rating_matrix, test_set, candidate_k_list):
    """
    【功能】为算法选出最优的近邻个数 K。思路是"藏起一部分已知评分当考题"，看哪个 K 预测这些考题最准。
           1. 数据的准备（main() 第 2、3 步已完成，本函数不用做）：全部评分随机分成两份，
              80% 是"已知评分"（训练集），用来建评分矩阵 rating_matrix；
              20% 是"隐藏评分"（测试集 test_set），没有填进评分矩阵，在矩阵中对应的格子是 0，相当于算法不知道这些评分。
           2. 计算相似度：调用 algorithm.compute_similarity(rating_matrix)，得到相似度矩阵。
              相似度只用已知评分计算，与 K 无关，只需计算一次。
           3. 对每个候选 K 预测考题：对 candidate_k_list 中的每个 K，遍历测试集的每一条记录（用户、电影、真实评分），
              调用前面已完成的 algorithm.predict_rating(rating_matrix, similarity_matrix, 用户下标, 电影下标, K) 得到预测评分。
              不需要另写预测算法：algorithm 是 user_cf 时就是函数 2，是 item_cf 时就是函数 4。
              测试集中的 user_id、movie_id 从 1 开始编号，作为矩阵下标时要减 1。
           4. 计算误差：把这个 K 下所有记录的预测评分与真实评分比较，计算均方根误差
              RMSE = √( Σ(真实评分 − 预测评分)² / 测试记录条数 )，记入 rmse_list。
           5. 选出最优 K：所有候选 K 都算完后，RMSE 最小的那个 K 就是最优 K。

    【输入】algorithm        : 算法模块 user_cf 或 item_cf。本函数只调用它的两个函数：
                                algorithm.compute_similarity(...) 和 algorithm.predict_rating(...)，
                              所以同一个函数两种算法都能用
           rating_matrix    : 用已知评分（训练集）建成的评分矩阵，形状 (943, 80)，隐藏评分对应的格子为 0
           test_set         : 隐藏评分（测试集），DataFrame，三列 user_id、movie_id、rating，rating 就是考题的标准答案
           candidate_k_list : 候选 K 的列表，如 [5, 10, 20, 30, 50, 80, 120, 200, 400]

    【输出】best_k    : 最优 K（int）
           rmse_list : 列表，rmse_list[j] 是 candidate_k_list[j] 对应的 RMSE（float）
    """
    # 相似度只用已知评分计算，与 K 无关，只需计算一次
    similarity_matrix = algorithm.compute_similarity(rating_matrix)
    rmse_list = []
    for k_neighbors in candidate_k_list:
        squared_errors = []
        for record in test_set.itertuples(index=False):
            # 测试集编号从 1 开始，矩阵下标从 0 开始，都要减 1
            predicted_rating = algorithm.predict_rating(
                rating_matrix, similarity_matrix,
                record.user_id - 1, record.movie_id - 1, k_neighbors)
            squared_errors.append((record.rating - predicted_rating) ** 2)
        rmse_list.append(float(np.sqrt(np.mean(squared_errors))))
    best_index = int(np.argmin(rmse_list))
    best_k = candidate_k_list[best_index]                            # RMSE 最小的 K 即最优 K
    return best_k, rmse_list


# ======================= 以下为协同过滤算法的测试用的方法 =======================
def recommend_top_n(algorithm, rating_matrix, similarity_matrix, user_index, k_neighbors, top_n=10):
    """
    【测试用 · 已写好】用你写的算法，真正给一个用户推荐电影（课件中的第 ⑤ 步"排序推荐"）。
    写好的 compute_similarity() 和 predict_rating() 最终就是在这里被用来做推荐的：
        对该用户没评过分的每部电影调用 predict_rating() 预测评分，再按预测评分从高到低排序，取前 top_n 部。
    main() 用它给 1 号用户做推荐测试；网页给"你"推荐电影时，调用的也是这个函数。

    【输入】algorithm         : 算法模块 user_cf 或 item_cf
           rating_matrix     : 评分矩阵（推荐时用全部评分）
           similarity_matrix : algorithm.compute_similarity(rating_matrix) 的结果
           user_index        : 要推荐的用户的行下标（从 0 开始）
           k_neighbors       : 近邻个数 K（用 choose_best_k() 选出的最优 K）
           top_n             : 推荐几部；为 None 时返回全部没看过的电影

    【输出】列表 [(电影下标, 预测评分), ...]，按预测评分从高到低排列
    """
    unseen_movies = np.where(rating_matrix[user_index] == 0)[0]          # 该用户没评过分的电影
    predictions = [(int(movie_index),
                    float(algorithm.predict_rating(rating_matrix, similarity_matrix, user_index, movie_index, k_neighbors)))
                   for movie_index in unseen_movies]                       # 逐部预测评分
    predictions.sort(key=lambda item: item[1], reverse=True)               # 预测评分从高到低排序
    return predictions if top_n is None else predictions[:top_n]


def main():
    # ---------- 第 1 步：读取数据 ----------
    movies = pd.read_csv(os.path.join(ROOT, "data", "movies.csv"), encoding="utf-8")
    ratings = pd.read_csv(os.path.join(ROOT, "data", "ratings.csv"))
    num_users, num_movies = ratings.user_id.max(), len(movies)
    print(f"第1步  读取数据：{num_users} 位用户，{num_movies} 部电影，{len(ratings)} 条评分")

    # ---------- 第 2 步：划分训练集和测试集（8 : 2）----------
    shuffled = ratings.sample(frac=1, random_state=42)
    cut = int(len(shuffled) * 0.8)
    train_set, test_set = shuffled[:cut], shuffled[cut:]
    print(f"第2步  训练集 {len(train_set)} 条，测试集 {len(test_set)} 条（测试集的分数先藏起来，用来检验预测准不准）")

    # ---------- 第 3 步：用训练集构建评分矩阵 ----------
    rating_matrix = build_matrix(train_set, num_users, num_movies)
    print(f"第3步  评分矩阵：{rating_matrix.shape[0]} 行（用户）× {rating_matrix.shape[1]} 列（电影），"
          f"有评分的格子只占 {(rating_matrix > 0).mean():.0%}")

    full_matrix = build_matrix(ratings, num_users, num_movies)   # 全部评分的矩阵，推荐测试时使用
    movie_names = dict(zip(movies.movie_id - 1, movies.name))

    best_k_of, curves = {}, {}
    for key, (name, algorithm) in ALGORITHMS.items():
        print(f"\n========== {name} ==========")
        # ---------- 第 4 步：计算相似度，在测试集上选出最优 K ----------
        try:
            t = time.time()
            best_k, rmse_list = choose_best_k(algorithm, rating_matrix, test_set, CANDIDATE_K[key])
        except NotImplementedError as e:
            print(f"  还没完成：{e}（先跳过）")
            continue
        print(f"第4步  选择近邻个数 K（用时 {time.time() - t:.1f} 秒）：")
        for k_neighbors, rmse in zip(CANDIDATE_K[key], rmse_list):
            print(f"         K = {k_neighbors:<4d} 测试集 RMSE = {float(rmse):.4f}")
        print(f"       → 最优 K = {best_k}，RMSE = {min(rmse_list):.4f}")
        best_k_of[key] = int(best_k)
        curves[key] = {"k": CANDIDATE_K[key], "rmse": [round(float(e), 4) for e in rmse_list], "best": int(best_k)}

        # 测试：用全部评分和最优 K，调用 recommend_top_n() 给 1 号用户推荐 5 部电影
        similarity_matrix = algorithm.compute_similarity(full_matrix)
        top5 = recommend_top_n(algorithm, full_matrix, similarity_matrix, user_index=0, k_neighbors=best_k, top_n=5)
        print("       测试 recommend_top_n()：给 1 号用户推荐的 5 部电影")
        for rank, (movie_index, score) in enumerate(top5, 1):
            print(f"         {rank}. 《{movie_names[movie_index]}》  预测评分 {score:.2f}")

    # ---------- 第 5 步：打开网页，给你推荐电影 ----------
    # 网页上打的分会保存到 data/my_ratings.json，"你"作为第 944 号用户加入全部评分数据，
    # 用两种算法（各自的最优 K）预测你对每部没看过的电影的评分，取最高的 10 部推荐给你。
    if not best_k_of:
        print("\n还没有算法完成第4步，网页暂时打不开。")
        return
    print("\n第5步  打开推荐网页（按 Ctrl+C 结束）")
    import web_server
    web_server.run(ALGORITHMS, best_k_of, curves, build_matrix, recommend_top_n, movies, ratings)


if __name__ == "__main__":
    main()
