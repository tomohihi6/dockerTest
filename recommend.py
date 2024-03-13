from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
import pandas as pd

data = Dataset.load_builtin("ml-100k")

trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

algo = SVD()
algo.fit(trainset)

predictions = algo.test(testset)

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n
top_n = get_top_n(predictions, n=10)

def get_top_n_df(top_n):
    top_n_df = []
    for uid, user_ratings in top_n.items():
        for iid, est in user_ratings:
            top_n_df.append([uid, iid, est])
    return pd.DataFrame(top_n_df, columns=['User', 'Item', 'Estimate'])
top_n_df = get_top_n_df(top_n)
print(top_n_df)