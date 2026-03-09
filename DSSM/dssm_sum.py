from torch import device
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from tqdm import tqdm
import random
import os
import sys
import jieba
import jieba.analyse

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from data_handler import DataPreprocessor

"""
Pytorch Dataset and Model
"""


class DSSMDataset(Dataset):
    """
    为了将数据送入PyTorch的DataLoader而自定义的Dataset类处理。
    负责在获取单个批次样本时拼装所需的所有图书特征。
    """

    def __init__(self, users, books, labels, books_df):
        self.users = users
        self.books = books
        self.labels = labels
        self.books_df = books_df  # 包含特征的DataFrame

        # 让图书按索引排列以确保能直接通过图书索引作为数值下标查询特征
        sorted_books = books_df.sort_index()

        # 提取数组，加快__getitem__中读取速度，替代原生Pandas iloc读取
        self.cat_array = sorted_books["category_idx"].values
        self.subcat_array = sorted_books["subcategory_idx"].values
        self.author_array = sorted_books["author_idx"].values
        self.pub_array = sorted_books["publisher_idx"].values
        self.pyear_array = sorted_books["pubyear_idx"].values
        self.tag_array = np.stack(sorted_books["tag_seq"].tolist())
        self.sum_array = np.stack(sorted_books["sum_seq"].tolist())

    def __len__(self):
        # 返回数据集样本总量
        return len(self.users)

    def __getitem__(self, idx):
        # 根据索引获取样本
        user = self.users[idx]
        book = self.books[idx]
        label = self.labels[idx]

        # 获取图书的特征
        cat = self.cat_array[book]
        subcat = self.subcat_array[book]
        author = self.author_array[book]
        pub = self.pub_array[book]
        pyear = self.pyear_array[book]
        tag_seq = self.tag_array[book]
        sum_seq = self.sum_array[book]

        # 转化为PyTorch tensor
        return {
            "user": torch.tensor(user, dtype=torch.long),
            "book": torch.tensor(book, dtype=torch.long),
            "cat": torch.tensor(cat, dtype=torch.long),
            "subcat": torch.tensor(subcat, dtype=torch.long),
            "author": torch.tensor(author, dtype=torch.long),
            "pub": torch.tensor(pub, dtype=torch.long),
            "pyear": torch.tensor(pyear, dtype=torch.long),
            "tag": torch.tensor(tag_seq, dtype=torch.long),
            "sum": torch.tensor(sum_seq, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
        }


class DSSM(nn.Module):
    """
    DSSM双塔模型网络
    分别用两个塔对User和Item进行Embedding及MLP变换后映射为向量,最后使用内积算分。
    """

    def __init__(
        self,
        num_users,
        num_books,
        num_cats,
        num_subcats,
        num_authors,
        num_publishers,
        num_pubyears,
        num_tags,
        num_words,
        embed_dim=64,
        hidden_dims=[256, 128],
    ):
        super(DSSM, self).__init__()

        # 用户塔
        # User ID的嵌入矩阵，将稀疏ID转为稠密向量(embed_dim维)
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        # User端的DNN
        self.user_mlp = nn.Sequential(
            nn.Dropout(p=0.2),  # 添加轻量Dropout防止死记硬背
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Dropout
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),  # 利用Tanh将得分规整化到一定空间区间
        )

        # 物品塔
        # Item包含基础的Item_ID Embedding
        self.item_embedding = nn.Embedding(num_books, embed_dim)

        # 以及一系列附属Side Information，降低维度作为辅助
        self.cat_embedding = nn.Embedding(num_cats, embed_dim)
        self.subcat_embedding = nn.Embedding(num_subcats, embed_dim // 2)
        self.author_embedding = nn.Embedding(num_authors, embed_dim)
        self.pub_embedding = nn.Embedding(num_publishers, embed_dim // 2)
        self.pyear_embedding = nn.Embedding(num_pubyears, embed_dim // 2)
        # Tag 序列表达(padding_idx表示忽略用0填充的不参与梯度更新计算)
        self.tag_embedding = nn.Embedding(num_tags, embed_dim // 2, padding_idx=0)
        self.word_embedding = nn.Embedding(num_words, embed_dim, padding_idx=0)

        # 6个 (embed_dim // 2) 维度的特征 + 2个 embed_dim 的特征 (book_id和sum)
        book_input_dim = embed_dim * 4 + (embed_dim // 2) * 4

        # Book端的DNN结构，映射到和User相同的向量维度hidden_dims[1]中
        self.book_mlp = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(book_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
        )

    def user_forward(self, user):
        """用户塔的前向传播,离线推理出用户的Embedding。"""
        u_emb = self.user_embedding(user)
        u_vector = self.user_mlp(u_emb)
        return u_vector

    def item_forward(self, book, cat, subcat, author, pub, pyear, tag, sum_seq):
        """物品塔的前向传播,计算出物品的Embedding。"""
        b_emb = self.item_embedding(book)
        c_emb = self.cat_embedding(cat)
        sub_emb = self.subcat_embedding(subcat)
        auth_emb = self.author_embedding(author)
        pub_emb = self.pub_embedding(pub)
        pyear_emb = self.pyear_embedding(pyear)

        # Tag是一个由ID组成的定长序列，这里直接通过取平均转为单一特征表示
        tag_emb = self.tag_embedding(tag).mean(dim=1)
        sum_emb = self.word_embedding(sum_seq).mean(dim=1)

        # 将所有Item特征在特征维度拼接在一起
        concat_emb = torch.cat(
            [b_emb, c_emb, sub_emb, auth_emb, pub_emb, pyear_emb, tag_emb, sum_emb],
            dim=1,
        )

        b_vector = self.book_mlp(concat_emb)
        return b_vector

    def forward(self, batch_data):
        """前向传播,计算用户和物品的向量并计算相似度得分。"""
        # 计算用户向量
        u_vec = self.user_forward(batch_data["user"])
        # 计算物品向量
        b_vec = self.item_forward(
            batch_data["book"],
            batch_data["cat"],
            batch_data["subcat"],
            batch_data["author"],
            batch_data["pub"],
            batch_data["pyear"],
            batch_data["tag"],
            batch_data["sum"],
        )
        # [Batch, dim] * [Batch, dim] 逐元素相乘后再按维度dim求和等于点积，计算相似度。
        # 打分越高代表用户对这个物品的偏好度越大。
        score = torch.sum(u_vec * b_vec, dim=1)
        return score


def train_model(
    model, data_handler, neg_ratio=2, batch_size=256, epochs=5, lr=0.001, device="cuda"
):
    """
    训练DSSM模型:优化器、Loss和动态加载数据
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # 自动处理sigmoid
    # 引入 weight_decay(L2正则化) 防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 引入动态学习率调度器：如果评价指标(如Recall)在 patience 轮内没有提升，就使学习率衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_recall = 0.0
    patience_counter = 0
    early_stop_patience = 8  # 如果连续 8 次 Epoch 都没有突破历史最高分，则触发早停

    for epoch in range(epochs):
        model.train()  # 开启训练模式
        total_loss = 0

        # 每个Epoch重新进行一轮动态负采样，保证负样本随迭代随机化。
        u_train, i_train, y_train = data_handler.get_train_samples(neg_ratio)
        train_dataset = DSSMDataset(u_train, i_train, y_train, data_handler.books)

        # 使用DataLoader构建数据加载器处理批量送入数据
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            for k, v in batch.items():
                batch[k] = v.to(device)

            optimizer.zero_grad()  # 清零梯度
            scores = model(batch)  # 模型前向传播
            loss = criterion(scores.view(-1), batch["label"].view(-1))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累加损失

        print("Epoch: ", epoch + 1, "Loss: ", total_loss / len(train_loader))

        # 获取当前Epoch的Recall得分
        current_recall = evaluate_recall(model, data_handler, Top_K=20, device=device)

        # 步进动态学习率调度器
        scheduler.step(current_recall)

        # Early Stopping 核心逻辑
        if current_recall > best_recall:
            best_recall = current_recall
            patience_counter = 0  # 重置耐心值

            print(f"创造新高！最优 Recall 更新为: {best_recall:.4f}")
        else:
            patience_counter += 1
            print(f"Recall 未提升 ({patience_counter}/{early_stop_patience})")
            if patience_counter >= early_stop_patience:
                print("Early Stopping")
                print(best_recall)
                break


def evaluate_recall(model, data_handler, Top_K=20, device="cuda"):
    """
    评估模型性能:计算Recall@K等指标
    """
    model.eval()  # 切换到评估模式
    book_df = data_handler.books
    print("Evaluating recall...")
    book_vectors = []
    book_ids = []

    # 构造一个Dataloader来批量处理book
    all_books_indices = book_df.index.tolist()
    batch_size = 512

    # 在评估中进行正向推导时不产生梯度记录以节省内存资源
    with torch.no_grad():
        for i in range(0, len(all_books_indices), batch_size):
            batch_idx = all_books_indices[i : i + batch_size]
            batch_df = book_df.loc[batch_idx]

            # 批量处理book
            books = torch.tensor(batch_idx, dtype=torch.long).to(device)
            cats = torch.tensor(batch_df["category_idx"].values, dtype=torch.long).to(
                device
            )
            subcats = torch.tensor(
                batch_df["subcategory_idx"].values, dtype=torch.long
            ).to(device)
            authors = torch.tensor(batch_df["author_idx"].values, dtype=torch.long).to(
                device
            )
            pubs = torch.tensor(batch_df["publisher_idx"].values, dtype=torch.long).to(
                device
            )
            pyears = torch.tensor(batch_df["pubyear_idx"].values, dtype=torch.long).to(
                device
            )

            tags = torch.tensor(
                np.stack(batch_df["tag_seq"].values), dtype=torch.long
            ).to(device)
            sums = torch.tensor(
                np.stack(batch_df["sum_seq"].values), dtype=torch.long
            ).to(device)

            vectors = model.item_forward(
                books, cats, subcats, authors, pubs, pyears, tags, sums
            )
            book_vectors.append(vectors.cpu().numpy())
            book_ids.extend(batch_idx)

    # 将所有Books通过item_forward提取稠密表示向量
    book_vectors = np.concatenate(book_vectors, axis=0)

    # 建立由于向量维度导致的映射偏移。以供"用户训练交互记录过滤"。
    book_id_to_col = {b_id: col_idx for col_idx, b_id in enumerate(book_ids)}

    print("Evaluating recall...")
    hits = 0  # TopK内的真实正样本命中次数
    total_true = 0  # 测试集中总体的相关物品数量
    pred_hits = 0  # 判定命中的总条目
    total_pred = 0  # 总共推荐的条目数量=Users*Top_K
    user_hit = 0  # 有多少个用户至少被命中推测了1个目标
    total_user = 0  # 参与评价的总用户数量
    ndcg_sum = 0.0  # 评价序的指标累计值

    test_users = list(data_handler.test_dict.keys())

    # 遍历测试集中的用户，计算推荐结果
    with torch.no_grad():
        for i in range(0, len(test_users), batch_size):
            batch_users = test_users[i : i + batch_size]
            u_input = torch.tensor(batch_users, dtype=torch.long).to(device)

            # 批量提取用户向量
            u_vectors = model.user_forward(u_input).cpu().numpy()

            # 计算用户向量与所有物品向量的点积，得到用户对所有物品的预测得分
            scores = np.dot(u_vectors, book_vectors.T)

            # 为当前批次中的每一个User统计TopK表现
            for idx, user_id in enumerate(batch_users):
                true_books = set(data_handler.test_dict[user_id])
                if len(true_books) == 0:
                    continue

                #  确保测试时不给用户推荐之前他们已经读过的(即被分配至训练集中的记录作为负无穷)
                train_books = data_handler.train_dict.get(user_id, [])
                train_cols = [
                    book_id_to_col[b] for b in train_books if b in book_id_to_col
                ]
                scores[idx, train_cols] = -np.inf  # 训练集中的记录作为负无穷

                # argpartition能保证前Top_K的数在前面但不保证这几个内部的有序，可用于快速过滤选取最大的Top-K个索引位置
                part_idx = np.argpartition(-scores[idx], Top_K)[:Top_K]
                # 提取这Top-K个真实分数进行准确从大到小排序
                part_scores = scores[idx, part_idx]
                sorted_rel_idx = np.argsort(-part_scores)
                # 反推回到真实对应的全局预估图书Book Index位
                sorted_idx = part_idx[sorted_rel_idx]

                # 通过前K个物品编号索引取得具体的预测图书列表(被推荐的结果表)
                pred_books = [book_ids[i] for i in sorted_idx]

                # 计算相关指标
                hit_count = len(true_books.intersection(pred_books))
                hits += hit_count
                total_true += len(true_books)
                pred_hits += hit_count
                total_pred += Top_K

                if hit_count > 0:
                    user_hit += 1
                total_user += 1

                dcg = 0.0  # 折损累积增益
                for rank, rec_book in enumerate(pred_books):
                    if rec_book in true_books:
                        dcg += 1.0 / np.log2(rank + 2)

                idcg = 0.0  # 理想情况下的DCG值
                max_hits = min(len(true_books), Top_K)
                for i in range(max_hits):
                    idcg += 1.0 / np.log2(i + 2)

                if idcg > 0:
                    ndcg_sum += dcg / idcg
    # 输出评估结果
    recall_value = 0.0
    if total_true > 0:
        recall_value = hits / total_true
        # 召回率：所有判定命中的物品占比所有实际产生交互记录的正例之比
        print(f"Recall@{Top_K}: {recall_value:.4f}")
        # NDCG：衡量排序准确率，越高说明靠前的对的越多
        print(f"NDCG@{Top_K}: {ndcg_sum/total_user:.4f}")
        # HitRate：用户命中率，即有多少用户至少被推荐了一个相关物品
        print(f"HitRate@{Top_K}: {user_hit/total_user:.4f}")
        # Precision：推荐出的总量中有多少个真实的被命中产生
        print(f"Precision@{Top_K}: {pred_hits/total_pred:.4f}")

    return recall_value


def set_seed(seed=42):
    """
    固定所有使用到的多源随机种子，以便代码环境执行的结果每次都可复现 (Reproducible)。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    set_seed(42)  # 固定随机种子

    # 构建相对文件目录寻找数据集路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    book_path = os.path.join(base_dir, "data", "book.csv")
    rating_path = os.path.join(base_dir, "data", "ratings.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # 初始化预处理与词表读取
    handler = DataPreprocessor(book_path, rating_path)
    handler.load_and_preprocess_data()
    handler.build_text_features(max_tag_len=5, max_sum_len=50)
    handler.train_test_split(0.2)  # 划分训练集和测试集

    # 初始化DSSM模型
    model = DSSM(
        num_users=handler.num_users,
        num_books=handler.num_books,
        num_cats=handler.num_categories,
        num_subcats=handler.num_subcats,
        num_authors=handler.num_authors,
        num_publishers=handler.num_publishers,
        num_pubyears=handler.num_pubyears,
        num_tags=len(handler.tag_vocab),
        num_words=len(handler.word_vocab),
        embed_dim=64,
        hidden_dims=[128, 64],
    )

    # 训练模型
    train_model(
        model, handler, neg_ratio=2, batch_size=256, epochs=50, lr=0.001, device=device
    )
