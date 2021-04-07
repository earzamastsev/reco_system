import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import Pool
import pickle

# load data
print('Load data...')
trans = pd.read_csv('transactions.csv')
prods = pd.read_csv('products.csv')
df = trans.merge(prods, on=['product_id'])

# preprocessing and optimisation
print('Preprocessing...')
int64 = df.select_dtypes('int64').columns
float64 = df.select_dtypes('float64').columns
df.days_since_prior_order = df.days_since_prior_order.fillna(9999)
df[int64] = df[int64].astype('int32')
df[float64] = df[float64].astype('int32')
df.sort_values(by=['user_id', 'order_number', 'add_to_cart_order'], ignore_index=True, inplace=True)
df['time'] = df.index.values
df['time'] = df['time'].astype('int32')

# calc frequency
aggr = df.groupby(['user_id','product_id']) \
        .agg({'order_id':'count'}) \
        .rename(columns={'order_id':'cnt'}) \
        .sort_values('cnt', ascending=False)

# calc top10 prods for cold start
top10 = df.groupby('product_id').agg({'order_id':'count'}).rename(columns={'order_id':'cnt'}).sort_values(by='cnt', ascending=False).head(10)
top10 = top10.reset_index().merge(prods[['product_id','product_name']])

def make_submission(data):
    sub = {"user_id":[],
               "product_id":[]}

    for block in data:
        for key, value in block.items():
            sub[key] += value

    sub = pd.DataFrame(sub)
    sub.to_csv('submission.csv', index=False)
    print(f'Submission writed to submission.csv, rows={sub.shape[0]}')


def make_predicition(data):
    print(f'Start block {data[0]} - {data[-1]}...')
    results = {"user_id":[],
           "product_id":[]}
    
    for user in data:
        prods = aggr.loc[user].head(10).index.to_list()
        results["user_id"].append(user)
        if len(prods)<10:
            tmp = prods + top10.product_id.to_list()[0:10-len(prods)]
            results["product_id"].append(" ".join([str(x) for x in tmp]))
        else:
            results["product_id"].append(" ".join([str(x) for x in prods]))
    print(f'Finish block {data[0]} - {data[-1]}...')
    return results

print('Run Threads...')
step = 10000
users_block = []
for i in range(0, len(df.user_id.unique()), step):
        users_block.append(df.user_id.unique()[i:i+step])
        
p = Pool(processes=7)
result = p.map(make_prediction,  users_block)
p.close()
p.join()

print('Finish', len(result))

pickle.dump(result, open('sub_sync.pkl', 'wb'))
make_submission(result)
