import pandas as pd

# 读取你的评论数据
df = pd.read_csv('data2/ratings.csv')

# 1. 基础统计
n_users = df['UserID'].nunique()
n_items = df['BookID'].nunique()
n_ratings = len(df)

print(f"📚 书籍数: {n_items}")
print(f" 用户数: {n_users}")
print(f"📝 评分数: {n_ratings}")

# 2. 关键指标：平均每个用户读了几本书？
avg_actions = n_ratings / n_users
print(f"📊 平均每人读过: {avg_actions:.4f} 本书")

# 3. 关键指标：有多少用户是“回头客”？
user_counts = df['UserID'].value_counts()
at_least_2 = user_counts[user_counts >= 2].count()
at_least_5 = user_counts[user_counts >= 5].count()

print(f"👥 读过 >= 2 本书的用户数: {at_least_2} (占比 {at_least_2/n_users:.2%})")
print(f"👥 读过 >= 5 本书的用户数: {at_least_5} (占比 {at_least_5/n_users:.2%})")

# 4. ItemCF 的生存底线
if avg_actions < 1.1:
    print("\n⚠️ 警告：数据极度稀疏！绝大多数用户只出现了一次。ItemCF 和 DIN 将无法运行。")
    print("👉 建议方案：不要爬新书了，去爬现有这 1140 本书的更多评论页！")
else:
    print("\n✅ 数据质量尚可，可以跑通模型，但推荐效果可能偏向热门书。")