import pandas as pd

print("正在合并所有评论数据...")

df1 = pd.read_csv('data/ratings_final.csv')
df2 = pd.read_csv('data/ratings_plus.csv')
df3 = pd.read_csv('data/ratings_plus_part2.csv')
df4 = pd.read_csv('data/ratings_plus_part3.csv')
# 2. 合并
df_all = pd.concat([df1, df2, df3, df4])

# 3. 去重 (关键步骤)
df_all.drop_duplicates(subset=['UserID', 'BookID'], keep='last', inplace=True)

# 4. 保存最终版
df_all.to_csv('data/ratings_ultimate.csv', index=False, encoding='utf-8-sig')

print(f"🎉 恭喜！最终数据集构建完成！")
print(f"📊 总交互数据量: {len(df_all)}")