import os
import sys
import numpy as np
import pandas as pd
import math
import random
from tqdm import tqdm
from collections import defaultdict

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from data_handler import DataPreprocessor


class ItemCFRecommender:
    def __init__(self, k_sim_items=30, n_rec_items=10):
        """
        :param k_sim_items: 计算相似度时,每个物品只保留最相似的K个物品
        :param n_rec_items: 最终给用户推荐的物品数量
        """
        self.k_sim_items = k_sim_items
        self.n_rec_items = n_rec_items
        self.item_sim_matrix = dict()

    def fit(self, train_dict, ratings_df):
        """
        基于训练集计算物品相似度矩阵(考虑分数)
        """
        # 建立 user-book 到 rating 的映射
        user_book_to_rating = dict(
            zip(
                zip(ratings_df["user_idx"], ratings_df["book_idx"]),
                ratings_df["Rating"],
            )
        )

        # 构建包含分数的 train_data
        self.train_data = defaultdict(dict)
        for user, items in train_dict.items():
            for item in items:
                self.train_data[user][item] = float(
                    user_book_to_rating.get((user, item), 1.0)
                )

        V = defaultdict(float)  # 记录 (物品i, 物品j) 同时出现的次数
        W = defaultdict(float)  # 记录 物品i 被多少个用户看过

        print("计算物品共现和权重...")
        for users, item_ratings in tqdm(self.train_data.items()):
            items = list(item_ratings.keys())

            for i in range(len(items)):
                item_i = items[i]
                rating_i = item_ratings[item_i]

                W[item_i] += rating_i * rating_i

                for j in range(i + 1, len(items)):
                    item_j = items[j]
                    rating_j = item_ratings[item_j]

                    product = rating_i * rating_j
                    pair_key = tuple(sorted((item_i, item_j)))
                    V[pair_key] += product

        self.item_sim_matrix = defaultdict(dict)

        print("正在归一化相似度...")
        for (i, j), dot_product in V.items():
            sim = dot_product / math.sqrt(W[i] * W[j])

            self.item_sim_matrix[i][j] = sim
            self.item_sim_matrix[j][i] = sim

        for item in self.item_sim_matrix:
            self.item_sim_matrix[item] = dict(
                sorted(
                    self.item_sim_matrix[item].items(), key=lambda x: x[1], reverse=True
                )[: self.k_sim_items]
            )

    def recommend(self, user, top_k=None):
        """
        为单个用户推荐图书
        原理：找到用户历史上喜欢的书，把这些书的相似书加权推荐出来
        """
        if top_k is None:
            top_k = self.n_rec_items

        rank = defaultdict(float)
        interacted_items = self.train_data.get(user, dict())

        # 遍历用户历史交互过的物品 i
        for i, r_ui in interacted_items.items():
            # 找到与 i 相似的物品 j
            if i not in self.item_sim_matrix:
                continue

            for j, wij in self.item_sim_matrix[i].items():
                if j in interacted_items:
                    continue  # 过滤掉用户已经看过的书

                # j 的分数 += 相似度(i, j) * 用户对i的兴趣
                rank[j] += wij * r_ui

        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]


def evaluate_recall(model, data_handler, Top_K=20):
    """
    使用 Recall@K, NDCG@K, HitRate@K 评估体系
    """
    print(f"Evaluating recall@{Top_K}...")
    hits = 0
    total_true = 0
    pred_hits = 0
    total_pred = 0
    user_hit = 0
    total_user = 0
    ndcg_sum = 0.0

    test_users = list(data_handler.test_dict.keys())

    for user_id in tqdm(test_users, desc="Evaluating"):
        true_books = set(data_handler.test_dict[user_id])
        if len(true_books) == 0:
            continue

        rec_list = model.recommend(user_id, top_k=Top_K)
        pred_books = [x[0] for x in rec_list]

        hit_count = len(true_books.intersection(pred_books))
        hits += hit_count
        total_true += len(true_books)
        pred_hits += hit_count
        total_pred += Top_K

        if hit_count > 0:
            user_hit += 1
        total_user += 1

        dcg = 0.0
        for rank, rec_book in enumerate(pred_books):
            if rec_book in true_books:
                dcg += 1.0 / np.log2(rank + 2)

        idcg = 0.0
        max_hits = min(len(true_books), Top_K)
        for i in range(max_hits):
            idcg += 1.0 / np.log2(i + 2)

        if idcg > 0:
            ndcg_sum += dcg / idcg

    if total_true > 0:
        recall_value = hits / total_true
        print(f"Recall@{Top_K}: {recall_value:.4f}")
        print(f"NDCG@{Top_K}: {ndcg_sum/total_user:.4f}")
        print(f"HitRate@{Top_K}: {user_hit/total_user:.4f}")
        print(f"Precision@{Top_K}: {pred_hits/total_pred:.4f}")

    return recall_value


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# --- 主程序入口 ---
if __name__ == "__main__":
    set_seed(42)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    books_file = os.path.join(base_dir, "data", "book.csv")
    ratings_file = os.path.join(base_dir, "data", "ratings.csv")

    # 数据预处理
    handler = DataPreprocessor(books_file, ratings_file)
    handler.load_and_preprocess_data()
    # 对于ItemCF，因为不用深度网络读取文本和分类序列，直接拆集即可，无需build_text_features
    handler.train_test_split(0.2)

    # 初始化ItemCFRecommender
    recommender = ItemCFRecommender(k_sim_items=30, n_rec_items=20)

    # 计算带分数的相似度
    recommender.fit(handler.train_dict, handler.ratings)

    # 评估模型
    evaluate_recall(recommender, handler, Top_K=20)
