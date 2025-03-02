import numpy as np
from tqdm import tqdm
import pandas as pd
import cvxpy as cp

def mse_reg(X: pd.DataFrame, y: pd.Series):
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    _, d = X_np.shape
    beta = cp.Variable(d)
    intercept = cp.Variable()

    residuals = y_np - (X_np @ beta + intercept)
    objective = cp.Minimize(cp.sum_squares(residuals))
    problem = cp.Problem(objective)
    problem.solve()

    return beta.value, intercept.value

def linear_trend(days_from_train: int, train: pd.DataFrame):

    lil_cols = ['total_orders', 'weekday_avg_sales']
    mid_cols = lil_cols + [f'lag_{i}' for i in list(range(days_from_train, 15))] \
        + [f'product_sales_{i}' for i in range(days_from_train, 15)]
    big_cols = mid_cols + [f'product_sales_{i}' for i in range(15, 22)] \
        + ['product_sales_28', 'product_sales_35', 'moving', 'week_moving_trend'] \
        + ['normed_week_mean', 'normed_week_median', 'week_trend'] \
        + [f'relative_price_{col}' for col in ['L2', 'L3', 'L4', 'kind']] \
        + [f'lag_{i}' for i in [21, 28, 35]]
    
    train = train.copy()
    train[big_cols + ['sales']] = train[big_cols + ['sales']].apply(np.sqrt)

    size = train[['unique_id', 'date']].groupby('unique_id', observed=True).count(
    ).reset_index().rename(columns={'date': 'n_many'})
    lil = size.loc[size['n_many'] < 40]
    mid = size.query('n_many < 60 and n_many >= 40')
    big = size.loc[size['n_many'] >= 60]

    datas = [(lil, lil_cols), (mid, mid_cols), (big, big_cols)]
    dict = {}
    for size, cols in datas:
        df = size.merge(train, how='inner', on='unique_id')[['unique_id', 'sales'] + cols]
        ids = df['unique_id'].unique().tolist()
        non_cols = big_cols.copy()
        for col in cols:
            non_cols.remove(col)
        for id in tqdm(ids):
            beta, intercept = mse_reg(df.loc[df['unique_id'] == id, cols], 
                                      df.loc[df['unique_id'] == id, 'sales'])
            dict[id] = {
                'unique_id': id,
                **{'coef_' + col: coef for col, coef in zip(cols, beta)},
                **{'coef_' + col: 0 for col in non_cols},
                'intercept': intercept
            }
        
    df = pd.DataFrame(dict).T
    df['unique_id'] = df['unique_id'].astype(int)
    df.to_csv(f'./trend_coefs_{days_from_train}.csv', index=False)
