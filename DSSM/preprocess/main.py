import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random


def preprocess_data(book_data='book.csv',rating_data='ratings.csv'):
	books=pd.read_csv(book_data)
	ratings=pd.read_csv(rating_data)

	books['BookID']=books['BookID'].astype(str)
	ratings['BookID']=ratings['BookID'].astype(str)

	merged_data=pd.merge(ratings,books,on='BookID',how='inner')
	merged_data['Tags']=merged_data['Tags'].fillna('')
	merged_data['Author']=merged_data['Author'].fillna('Unknown')
	merged_data['Publisher']=merged_data['Publisher'].fillna('Unknown')


