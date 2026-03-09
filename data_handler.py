import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from tqdm import tqdm
import random
import jieba
import jieba.analyse


class DataPreprocessor:
    """
    数据预处理类,用于加载用户评分数据和图书元数据,处理缺失值,
    进行标签编码Label Encoding,并划分训练集与测试集。
    """

    def __init__(self, book_path, rating_path):
        self.book_path = book_path
        self.rating_path = rating_path

        # 初始化LabelEncoder
        self.user_encoder = LabelEncoder()
        self.book_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.subcategory_encoder = LabelEncoder()
        self.author_encoder = LabelEncoder()
        self.publisher_encoder = LabelEncoder()
        self.pubyear_encoder = LabelEncoder()

        # 初始化Tag及Summary词汇表
        self.tag_vocab = {"<PAD>": 0, "<UNK>": 1}
        self.word_vocab = {"<PAD>": 0, "<UNK>": 1}

    def load_and_preprocess_data(self):
        """
        加载并预处理数据,处理缺失值和过滤交互过少的用户/图书
        """
        # 读取评分和图书数据
        ratings = pd.read_csv(
            self.rating_path, encoding="utf-8-sig", dtype={"UserID": str, "Rating": int}
        )
        books = pd.read_csv(self.book_path, encoding="utf-8-sig", low_memory=False)

        # 填充图书数据中的缺失值
        books["Tags"] = books["Tags"].fillna("")
        books["Summary"] = books["Summary"].fillna("")
        books["Category"] = books["Category"].fillna("Unknown")
        books["SubCategory"] = books["SubCategory"].fillna("Unknown")
        books["Author"] = books["Author"].fillna("Unknown")
        books["Publisher"] = books["Publisher"].fillna("Unknown")
        books["PubYear"] = books["PubYear"].fillna("Unknown").astype(str)

        print("原始评分记录数:", len(ratings))
        # 仅保留评分为正数（产生正向交互）的记录
        ratings = ratings[ratings["Rating"] > 0]

        # 统计每个用户和图书的交互次数
        book_cnt = ratings["BookID"].value_counts()
        user_cnt = ratings["UserID"].value_counts()

        # 过滤交互次数少于5的用户和图书,降低数据稀疏性
        valid_books = book_cnt[book_cnt >= 5].index
        valid_users = user_cnt[user_cnt >= 5].index

        ratings = ratings[ratings["BookID"].isin(valid_books)]
        ratings = ratings[ratings["UserID"].isin(valid_users)]

        books = books[books["BookID"].isin(ratings["BookID"].unique())]

        print("清洗后评分记录数:", len(ratings))
        print(
            "唯一用户数:",
            ratings["UserID"].nunique(),
            "  唯一图书数:",
            ratings["BookID"].nunique(),
        )

        # 对用户和图书ID进行标签编码
        ratings["user_idx"] = self.user_encoder.fit_transform(ratings["UserID"])
        books["book_idx"] = self.book_encoder.fit_transform(books["BookID"])

        # 将图书ID映射到对应的索引
        book_id_to_idx = dict(zip(books["BookID"], books["book_idx"]))
        ratings["book_idx"] = ratings["BookID"].map(book_id_to_idx)

        # cat sub_cat pub_year author publisher
        books["category_idx"] = self.category_encoder.fit_transform(books["Category"])
        books["subcategory_idx"] = self.subcategory_encoder.fit_transform(
            books["SubCategory"]
        )
        books["author_idx"] = self.author_encoder.fit_transform(books["Author"])
        books["publisher_idx"] = self.publisher_encoder.fit_transform(
            books["Publisher"]
        )
        books["pubyear_idx"] = self.pubyear_encoder.fit_transform(books["PubYear"])

        self.ratings = ratings
        self.books = books.set_index("book_idx")

        # 统计经过编码后各特征的类别数量（用于定义Embedding层的维度）
        self.num_users = len(self.user_encoder.classes_)
        self.num_books = len(self.book_encoder.classes_)
        self.num_categories = len(self.category_encoder.classes_)
        self.num_subcats = len(self.subcategory_encoder.classes_)
        self.num_authors = len(self.author_encoder.classes_)
        self.num_publishers = len(self.publisher_encoder.classes_)
        self.num_pubyears = len(self.pubyear_encoder.classes_)

        return ratings

    def build_text_features(self, max_tag_len=5, max_sum_len=30):
        """
        构建tag和Summary的索引序列。输入为以'|'分割的字符串,输出为定长的整型列表。
        """
        tag_seqs = []

        for tags in self.books["Tags"]:
            # 分解Tag,并限制最大数量
            tag_list = tags.split("|")[:max_tag_len]
            seq = []
            for tag in tag_list:
                # 动态构建Tag词表并转化成编号
                if tag not in self.tag_vocab:
                    self.tag_vocab[tag] = len(self.tag_vocab)
                seq.append(self.tag_vocab[tag])

            # 填充或截断序列至固定长度
            if len(seq) > max_tag_len:
                seq = seq[:max_tag_len]
            else:
                seq = seq + [0] * (max_tag_len - len(seq))
            tag_seqs.append(seq)

        print(f"Tag 词表大小: {len(self.tag_vocab)}")

        sum_seqs = []
        for summary in self.books["Summary"]:
            words = jieba.analyse.extract_tags(str(summary), topK=max_sum_len)
            seq = []
            for word in words:
                if word not in self.word_vocab:
                    self.word_vocab[word] = len(self.word_vocab)
                seq.append(self.word_vocab[word])
            if len(seq) > max_sum_len:
                seq = seq[:max_sum_len]
            else:
                seq = seq + [0] * (max_sum_len - len(seq))
            sum_seqs.append(seq)

        print(f"Summary 词表大小: {len(self.word_vocab)}")

        # 将转换结果加入到属性中
        self.books["tag_seq"] = tag_seqs
        self.books["sum_seq"] = sum_seqs
        self.max_tag_len = max_tag_len
        self.max_sum_len = max_sum_len

    def train_test_split(self, test_size=0.2):
        """
        拆分训练集和测试集。
        """
        train_dict = defaultdict(list)
        test_dict = defaultdict(list)

        # 按用户对他们的历史交互图书进行分组聚合
        user_grouped = self.ratings.groupby("user_idx")["book_idx"].apply(list)

        for user, books in tqdm(user_grouped.items(), desc="Splitting"):
            random.shuffle(books)  # 打乱顺序
            split_idx = int(len(books) * (1 - test_size))  # 计算切分点

            train_books = books[:split_idx]
            test_books = books[split_idx:]

            # 将拆分好的列表保存到以用户索引为key的字典中
            if train_books:
                train_dict[user] = train_books

            if test_books:
                test_dict[user] = test_books

        self.train_dict = train_dict
        self.test_dict = test_dict
        print("训练集用户数:", len(train_dict), "测试集用户数", len(test_dict))

    def get_train_samples(self, neg_ratio=4):
        """
        生成训练所需的(User,Book,Label)三元组,包含动态负采样

        """
        users_list, books_list, labels_list = [], [], []

        for user, pos_books in self.train_dict.items():
            pos_set = set(pos_books)
            for book in pos_books:
                # 记录一个正样本
                users_list.append(user)
                books_list.append(book)
                labels_list.append(1.0)
                # 为每个正样本采neg_ratio个负样本
                for _ in range(neg_ratio):
                    while True:
                        # 随机抽样
                        neg_book = random.randint(0, self.num_books - 1)
                        # 确保负样本不与正样本重复
                        if neg_book not in pos_set:
                            break
                    users_list.append(user)
                    books_list.append(neg_book)
                    labels_list.append(0.0)
        return np.array(users_list), np.array(books_list), np.array(labels_list)


if __name__ == "__main__":
    book_path = "data/book.csv"
    rating_path = "data/ratings.csv"
    handler = DataPreprocessor(book_path, rating_path)
    handler.load_and_preprocess_data()
