import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import winsound
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from settings import HEADERS

# --- 配置区 ---
INPUT_TASK_FILE = 'book_ids.csv'
OUTPUT_BOOKS = 'data/book.csv'
OUTPUT_RATINGS = 'data/ratings.csv'

# 多线程配置
MAX_WORKERS = 3

# 线程锁
save_lock = threading.Lock()  # 用于文件写入互斥
cookie_lock = threading.Lock()  # 用于更新 Cookie 互斥
print_lock = threading.Lock()  # 用于打印互斥，防止乱码


def safe_print(msg):
	"""线程安全的打印"""
	with print_lock:
		print(msg)


def safe_text(soup_element):
	return soup_element.text.strip() if soup_element else ""


def update_cookie():
	"""
	线程安全的 Cookie 更新逻辑。
	如果一个线程正在更新，其他线程会等待，并直接使用更新后的 Cookie。
	"""
	global HEADERS

	# 尝试获取锁，如果获取不到（说明有人正在更新），则阻塞等待
	with cookie_lock:
		# 再次检查（双重检查锁定）：可能在等待锁的时候，前一个线程已经更新好了
		# 这里做一个简单的测试，如果外部能检测到 cookie 变了最好，
		# 但为了简单，我们还是让用户确认或直接提示。

		safe_print("\n" + "=" * 60)
		safe_print(f"🛑 线程 {threading.current_thread().name} 触发豆瓣反爬虫！")
		safe_print("   ⚠️ 注意：如果刚才已经输入过新 Cookie，直接按回车即可复用。")
		safe_print("=" * 60)

		try:
			winsound.Beep(1000, 1000)
		except:
			pass

		new_cookie = input(
			f"👉 线程 {threading.current_thread().name} 请求新 Cookie (输入新值或回车复用, q退出): ").strip()

		if new_cookie.lower() == 'q':
			os._exit(0)  # 强制退出所有线程
			return False

		if new_cookie:
			HEADERS['Cookie'] = new_cookie
			safe_print("Cookie 已更新！")
		else:
			safe_print("尝试复用现有 Cookie...")

		time.sleep(3)  # 给一点缓冲时间
		return True


def get_html_with_retry(url, max_retries = 3):
	global HEADERS
	for attempt in range(max_retries):
		try:
			# 随机延时，多线程下这里的延时很重要
			time.sleep(random.uniform(2, 5))

			response = requests.get(url, headers = HEADERS, timeout = 10)

			# 反爬检测逻辑
			is_redirected = "/misc/sorry" in response.url
			has_robot_text = "机器人程序" in response.text or "点击证明" in response.text
			is_forbidden = response.status_code == 403

			if is_redirected or has_robot_text or is_forbidden:
				safe_print(f"[线程 {threading.current_thread().name}] 第 {attempt + 1} 次被拦截: {url}")
				if not update_cookie(): return None
				continue

			if response.status_code == 200:
				return BeautifulSoup(response.text, 'lxml')
			elif response.status_code == 404:
				safe_print(f" 404 Not Found: {url}")
				return None
		except Exception as e:
			safe_print(f" 网络错误: {e}")
			time.sleep(2)
	return None


# --- 核心处理逻辑 (与原版基本一致，微调返回值) ---

def process_one_book(book_id, category, sub_tag):
	url = f"https://book.douban.com/subject/{book_id}/"
	soup = get_html_with_retry(url)

	if not soup or not soup.select_one('h1 span'):
		return None, None

	meta = {'DoubanID': book_id, 'Category': category, 'SubCategory': sub_tag}

	try:
		# 1. 解析书籍元数据
		meta['Title'] = safe_text(soup.select_one('h1 span'))
		rating_tag = soup.select_one('strong.ll.rating_num')
		meta['Rating'] = safe_text(rating_tag) if safe_text(rating_tag) else "0.0"

		info_div = soup.select_one('#info')
		if info_div:
			text_full = info_div.get_text().replace('\n', ' ').replace('\xa0', ' ')

			# 作者
			author_span = info_div.find('span', string = re.compile(r'作者'))
			if author_span:
				next_a = author_span.find_next('a')
				meta['Author'] = next_a.text.strip().replace('\n', '').replace(' ',
																			   '') if next_a else author_span.next_sibling.strip()

			# 出版社
			pub_match = re.search(r'出版社:\s*([^\s]+)', text_full)
			if pub_match: meta['Publisher'] = pub_match.group(1)

			# 出版年
			year_match = re.search(r'出版年:\s*(\d{4})', text_full)
			if year_match: meta['PubYear'] = year_match.group(1)

		# 标签
		web_tags = [t.text.strip() for t in soup.select('#db-tags-section a.tag')]
		if not web_tags: web_tags = [t.text.strip() for t in soup.select('div.indent span a.tag')]
		meta['Tags'] = "|".join(list(set([category, sub_tag] + web_tags)))

		# 简介
		summary_full = soup.select_one('div.intro span.all.hidden')
		summary_short = soup.select_one('div.intro span.short')
		text = summary_full.get_text() if summary_full else (
			summary_short.get_text() if summary_short else safe_text(soup.select_one('div.intro')))
		meta['Summary'] = text.replace('...(展开全部)', '').replace('\n', '').strip()

		# 封面
		meta['CoverURL'] = soup.select_one('div#mainpic img')['src'] if soup.select_one('div#mainpic img') else ""

	except Exception as e:
		safe_print(f"解析元数据异常 {book_id}: {e}")
		return None, None

	# 2. 解析短评 (前5页)
	comments = []
	# 限制评论页数，减少请求量防止封号
	for page in range(5):
		c_url = f"https://book.douban.com/subject/{book_id}/comments/?status=P&limit=20&sort=new_score&start={page * 20}"
		c_soup = get_html_with_retry(c_url)
		if not c_soup: break

		items = c_soup.select('.comment-item')
		if not items: break

		for item in items:
			try:
				user_link = item.select_one('.avatar a')['href']
				user_id = user_link.strip('/').split('/')[-1]
				rating_span = item.select_one('.comment-info span.user-stars')

				rating_val = 0
				if rating_span:
					classes = rating_span.get('class')
					star_class = next((c for c in classes if 'allstar' in c), None)
					if star_class:
						rating_val = int(re.findall(r'\d+', star_class)[0]) / 10

				if rating_val > 0:
					comments.append({
						'UserID': user_id,
						'BookID': book_id,
						'Rating': rating_val,
						'Category': category
					})
			except:
				continue

	return meta, comments


def thread_worker(task_data):
	"""
	线程工作函数：执行抓取并立即保存
	"""
	bid = task_data['BookID']
	cat = task_data['Category']
	sub = task_data['SubCategory']

	try:
		meta, comments = process_one_book(bid, cat, sub)

		if meta:
			# 获取锁，写入文件
			with save_lock:
				# 写入 Books
				df_book = pd.DataFrame([meta])
				header_book = not os.path.exists(OUTPUT_BOOKS)
				df_book.to_csv(OUTPUT_BOOKS, mode = 'a', index = False, header = header_book, encoding = 'utf-8-sig')

				# 写入 Ratings
				if comments:
					df_ratings = pd.DataFrame(comments)
					header_rating = not os.path.exists(OUTPUT_RATINGS)
					df_ratings.to_csv(OUTPUT_RATINGS, mode = 'a', index = False, header = header_rating,
									  encoding = 'utf-8-sig')

			return f"✅ 成功: {bid} 《{meta['Title']}》 (评论: {len(comments)})"
		else:
			return f"⚠️ 失败: {bid} (无数据)"

	except Exception as e:
		return f"❌ 异常: {bid} - {e}"


# --- 主程序 ---
if __name__ == "__main__":
	if not os.path.exists(INPUT_TASK_FILE):
		print(f"错误：找不到任务文件 {INPUT_TASK_FILE}")
		exit()

	# 1. 读取任务列表
	task_df = pd.read_csv(INPUT_TASK_FILE)
	task_df['BookID'] = task_df['BookID'].astype(str)

	total_tasks = len(task_df)

	# 2. 检查已完成的任务 (断点续传)
	completed_ids = set()
	if os.path.exists(OUTPUT_BOOKS):
		try:
			existing = pd.read_csv(OUTPUT_BOOKS, usecols = ['DoubanID'])
			completed_ids = set(existing['DoubanID'].astype(str))
			print(f"检测到已存在数据: {len(completed_ids)} 本。")
		except:
			pass

	# 3. 准备任务队列
	tasks_to_run = []
	for index, row in task_df.iterrows():
		if str(row['BookID']) not in completed_ids:
			tasks_to_run.append(row)

	print(f"剩余任务: {len(tasks_to_run)} 本，开启线程数: {MAX_WORKERS}")
	print("=" * 60)

	# 4. 开启线程池
	with ThreadPoolExecutor(max_workers = MAX_WORKERS) as executor:
		# 提交所有任务
		future_to_book = {executor.submit(thread_worker, row): row['BookID'] for row in tasks_to_run}

		count = 0
		success_count = 0

		try:
			for future in as_completed(future_to_book):
				book_id = future_to_book[future]
				try:
					result = future.result()
					count += 1

					# 打印进度
					safe_print(f"[{count}/{len(tasks_to_run)}] {result}")

					if "成功" in result:
						success_count += 1

				except Exception as exc:
					safe_print(f"任务 {book_id} 抛出异常: {exc}")

		except KeyboardInterrupt:
			safe_print("\n🛑 用户手动停止！正在等待当前线程完成...")
			executor.shutdown(wait = False)

	print(f"\n🎉 全部完成！本次新增抓取: {success_count} 本")