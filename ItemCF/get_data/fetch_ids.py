import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import winsound
import os
from settings import HEADERS

# --- 配置区 ---
INTERMEDIATE_FILE = 'book_ids.csv'

# 确保目录存在
if not os.path.exists('data1'):
	os.makedirs('data1')

CRAWL_CONFIG = {
	"文学": ["小说", "日本文学", "名著", "散文", "村上春树"],
	"流行": ["悬疑", "推理", "科幻", "东野圭吾", "三体", "金庸"],
	"文化": ["历史", "心理学", "哲学", "传记"],
	"科技": ["互联网", "编程", "算法", "人工智能"],
	"经管": ["经济学", "投资", "管理"]
}


def update_cookie():
	print("\n" + "=" * 60)
	print("豆瓣反爬虫触发！(获取ID阶段)")
	try:
		winsound.Beep(1000, 1000)
	except:
		pass
	print("   1. 请在浏览器打开刚才的链接进行验证。")
	print("   2. 验证完成后，F12 获取最新 Cookie。")
	print("=" * 60)
	new_cookie = input("👉 请输入新 Cookie (回车继续，输入 q 退出): ").strip()
	if new_cookie.lower() == 'q': return False
	if new_cookie:
		HEADERS['Cookie'] = new_cookie
		time.sleep(3)
	return True


def get_html_with_retry(url, max_retries = 3):
	global HEADERS
	for attempt in range(max_retries):
		try:
			time.sleep(random.uniform(2, 4))  # 获取列表页稍微快一点
			response = requests.get(url, headers = HEADERS, timeout = 10)

			# 简单的反爬判断
			if "/misc/sorry" in response.url or response.status_code == 403:
				print(f"[第 {attempt + 1} 次] 被拦截: {url}")
				if not update_cookie(): return None
				continue

			if response.status_code == 200:
				return BeautifulSoup(response.text, 'lxml')
		except Exception as e:
			print(f" 网络错误: {e}")
			time.sleep(2)
	return None


def get_book_ids_by_config(category, sub_category, max_pages = 3):
	"""抓取特定标签下的书籍ID"""
	tasks = []
	base_url = "https://book.douban.com/tag/" + sub_category
	print(f"[正在搜索] 大类:{category} | 标签:{sub_category}")

	for page in range(max_pages):
		url = f"{base_url}?start={page * 20}&type=T"
		soup = get_html_with_retry(url)
		if not soup: break

		items = soup.select('ul.subject-list li.subject-item')
		if not items:
			print(f"第 {page + 1} 页无数据或解析失败")
			break

		count_in_page = 0
		for item in items:
			try:
				link = item.select_one('.info h2 a')['href']
				bid = link.strip('/').split('/')[-1]
				tasks.append({
					'BookID': bid,
					'Category': category,
					'SubCategory': sub_category
				})
				count_in_page += 1
			except:
				continue
		print(f"第 {page + 1} 页获取到 {count_in_page} 本")

	return tasks


# --- 主程序 ---
if __name__ == "__main__":
	all_tasks = []
	print("开始第一阶段：获取图书ID列表...")

	# 1. 遍历配置抓取
	for cat, sub_tags in CRAWL_CONFIG.items():
		for sub in sub_tags:
			tasks = get_book_ids_by_config(cat, sub, max_pages = 10)
			all_tasks.extend(tasks)

	# 2. 转换为 DataFrame 并去重
	if all_tasks:
		df = pd.DataFrame(all_tasks)
		# 根据 BookID 去重，保留第一个出现的分类信息
		df_unique = df.drop_duplicates(subset = ['BookID'])

		# 3. 保存
		df_unique.to_csv(INTERMEDIATE_FILE, index = False, encoding = 'utf-8-sig')
		print(f"\n✅ ID列表获取完成！")
		print(f"   - 原始抓取: {len(df)} 条")
		print(f"   - 去重后: {len(df_unique)} 条")
		print(f"   - 任务文件已保存至: {INTERMEDIATE_FILE}")
	else:
		print("\n⚠️ 未获取到任何数据，请检查网络或Cookie。")