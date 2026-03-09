import pandas as pd
import os
import jieba


def convert_to_recbole():
    print("开始读取原始数据...")
    books_df = pd.read_csv("dataset/book.csv")
    ratings_df = pd.read_csv("dataset/ratings.csv")

    dataset_name = "book_recipe"
    output_dir = f"dataset/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    print("正在生成 .inter 交互记录文件...")
    # ratings_df 包含 UserID,BookID,Rating, Category
    # 我们需要 user_id:token, item_id:token, rating:float
    inter_df = ratings_df[["UserID", "BookID", "Rating"]].copy()
    inter_df.columns = ["user_id:token", "item_id:token", "rating:float"]

    inter_path = os.path.join(output_dir, f"{dataset_name}.inter")
    inter_df.to_csv(inter_path, sep="\t", index=False, encoding="utf-8")
    print(f".inter 文件已保存到: {inter_path}")

    print("正在生成 .item 物品特征文件...")
    # books_df 包含 BookID,Category,SubCategory,Title,Rating,Author,Publisher,PubYear,Tags,Summary,CoverURL
    # 我们可以选取一些特征作为物品的 side_info
    # 需要处理一些可能包含tab或者缺失值的列
    item_df = books_df[
        [
            "BookID",
            "Category",
            "SubCategory",
            "Author",
            "Publisher",
            "PubYear",
            "Summary",
        ]
    ].copy()
    item_df.columns = [
        "item_id:token",
        "class:token",
        "sub_class:token",
        "author:token",
        "publisher:token",
        "pub_year:token",
        "summary:token_seq",
    ]

    # 将缺失值填充为特定的 token，比如 'unknown'
    item_df = item_df.fillna("unknown")

    # 确保没有换行符或制表符，否则会破坏 RecBole 的读取
    for col in item_df.columns:
        if item_df[col].dtype == "O":  # 如果是字符串类型
            item_df[col] = (
                item_df[col].astype(str).str.replace("\t", " ").str.replace("\n", " ")
            )

    # 使用jieba对summary进行分词，用空格连接（RecBole token_seq 默认分隔符）
    def segment_text(text):
        if text == "unknown" or not isinstance(text, str):
            return str(text)
        return " ".join(jieba.cut(text))

    print("正在使用jieba分词处理 Summary 列（可能需要十几秒）...")
    item_df["summary:token_seq"] = item_df["summary:token_seq"].apply(segment_text)

    item_path = os.path.join(output_dir, f"{dataset_name}.item")
    item_df.to_csv(item_path, sep="\t", index=False, encoding="utf-8")
    print(f".item 文件已保存到: {item_path}")

    print("\n转换完成！数据存放在: ", output_dir)
    print("你现在可以使用 RecBole 运行基准测试了。")


if __name__ == "__main__":
    convert_to_recbole()
