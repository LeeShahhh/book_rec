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


class ItemCF:
    def __init__(self, k_sim_items=30, n_rec_items=10):
        """
        :param k_sim_items: 计算相似度时,每个物品只保留最相似的K个物品
        :param n_rec_items: 最终给用户推荐的物品数量
        """
        self.k_sim_items = k_sim_items
        self.n_rec_items = n_rec_items
        self.item_sim_matrix = dict()

    def fit(self, train_dict):
        """
        基于训练集计算物品相似度矩阵(不考虑具体评分数值,基于共现计算)
        """
        self.train_dict = train_dict

        V = defaultdict(int)
        W = defaultdict(int)

        print("计算物品共现次数:")
        for user, items in tqdm(self.train_dict.items()):
            for i in range(len(items)):
                u_i = items[i]
                W[u_i] += 1
                for j in range(i + 1, len(items)):
                    u_j = items[j]
                    if u_j == u_i:
                        continue
                    V[tuple(sorted((u_i, u_j)))] += 1

        self.item_sim_matrix = defaultdict(dict)

        print("归一化截断计算相似度:")
        for (i, j), count in V.items():
            sim = count / math.sqrt(W[i] * W[j])
            self.item_sim_matrix[i][j] = sim
            self.item_sim_matrix[j][i] = sim

        # 仅保留最相似的K个物品
        for item in self.item_sim_matrix:
            self.item_sim_matrix[item] = dict(
                sorted(
                    self.item_sim_matrix[item].items(), key=lambda x: x[1], reverse=True
                )[: self.k_sim_items]
            )

    def recommend(self, user, top_k=None):
        """
        为用户user推荐图书,返回 (item, score) 列表
        """
        if top_k == None:
            top_k = self.n_rec_items

        rank = defaultdict(float)
        interacted_items = set(self.train_dict.get(user, []))

        for i in interacted_items:
            if i not in self.item_sim_matrix:
                continue

            for j, wij in self.item_sim_matrix[i].items():
                if j in interacted_items:
                    continue

                rank[j] += wij

        return sorted(rank.items(), key=lambda x: x[1], reverse=True)[:top_k]


def evaluate_recall(model, data_handler, Top_K=20):
    """
    使用 Recall@K,NDCG@K,HitRate@K,Precision@K 评估评估体系
    """
    print(f"Evaluating recall@{Top_K}:")
    hits = 0  # TopK内的真实正样本命中次数
    total_true = 0  # 测试集中总体的相关物品数量
    pred_hits = 0  # 判定命中的总条目
    total_pred = 0  # 总共推荐的条目数量=Users*Top_K
    user_hit = 0  # 有多少个用户至少被命中推测了1个目标
    total_user = 0  # 参与评价的总用户数量
    ndcg_sum = 0.0  # 评价序的指标累计值

    test_users = list(data_handler.test_dict.keys())

    for user_id in tqdm(test_users, desc="Evaluating"):
        true_books = set(data_handler.test_dict[user_id])
        if len(true_books) == 0:
            continue

        # 使用模型获取 TopK 推荐
        rec_list = model.recommend(user_id, top_k=Top_K)
        pred_books = [x[0] for x in rec_list]

        # 计算相关指标
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

    # 输出评估结果
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


if __name__ == "__main__":
    set_seed(42)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    book_path = os.path.join(base_dir, "data", "book.csv")
    rating_path = os.path.join(base_dir, "data", "ratings.csv")

    # 数据预处理
    handler = DataPreprocessor(book_path, rating_path)
    handler.load_and_preprocess_data()
    # 对于ItemCF，因为不用深度网络读取文本和分类序列，直接拆集即可，无需build_text_features
    handler.train_test_split(0.2)

    # 初始化ItemCF
    recaller = ItemCF(k_sim_items=30, n_rec_items=20)

    # 计算相似度
    recaller.fit(handler.train_dict)

    # 评估召回模型
    evaluate_recall(recaller, handler, Top_K=20)
