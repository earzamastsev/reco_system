import numpy as np
import pandas as pd
from tqdm import tqdm
from surprise import Dataset
from surprise import Reader
from surprise import SVD, SVDpp
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import pickle

def create_submission(data):
    results = {"user_id":[],
               "product_id":[]}
    for user, items in tqdm(data.items()):
        results["user_id"].append(user)
        results["product_id"].append(" ".join([str(x[0]) for x in items]))
    sub = pd.DataFrame(results)
    sub.to_csv('submission.csv', index=False)
    print(f'Save results: submission.csv\n, num records:" {sub.shape}\nExample record: {sub.iloc[1]}')
        


trans = pd.read_csv('transactions.csv')
prods = pd.read_csv('products.csv')

users = trans.user_id.unique()
items = prods.product_id.unique()

aggr = trans.groupby(['user_id','product_id']).agg({'user_id':'count'}).rename(columns={'user_id':'cnt'}).sort_values('cnt', ascending=False)
aggr['ranking'] = aggr.rank(pct=True)*5
aggr['ranking'] = aggr['ranking'].apply(round)
agg_rank = aggr.reset_index()
final_df = trans.merge(agg_rank, on=["user_id","product_id"])[["user_id","product_id", "ranking", "order_id"]]
final_df.columns = ["user", "item", "label", "time"]

data = final_df.groupby(['user', 'item']).agg({'label':'mean'}).reset_index()
reader = Reader(rating_scale=(1, 5)) # Зададим разброс оценок
data = Dataset.load_from_df(data, reader)
trainset = data.build_full_trainset()
testset = trainset.build_testset()
algo = SVD(n_factors = 20, n_epochs = 10, lr_all = 0.005, reg_all = 0.4, verbose=True)
algo.fit(trainset)

predictions = algo.test(testset)
acc = accuracy.rmse(predictions)
print(acc)

from collections import defaultdict

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n(predictions, n=10)
try:
    pickle.dump(top_n, open('top_n.pkl','wb'))
except:
    print('error')

print(top_n[1])

create_submission(top_n)

