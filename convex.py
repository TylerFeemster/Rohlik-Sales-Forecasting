import pandas as pd
import cvxpy as cp
import numpy as np
from preprocessing import get_calendar

def get_dataframe(city: str):
    df = pd.read_csv('./data/sales_train.csv').dropna()[['unique_id', 'date', 'sales', 'warehouse', 'total_orders']]
    df = df.dropna()
    df['city'] = df['warehouse'].apply(lambda x: x.split('_')[0])
    df = df.loc[df['city'] == city]
    df = df.drop(['city'], axis=1)
    
    # weights
    weights = pd.read_csv('./data/test_weights.csv')
    df = df.merge(weights, how='inner', on='unique_id')

    # basic features
    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[df['date'] >= '08-01-2020']

    df['dayofweek'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year

    # df feature engineering
    week_mean = df[['sales', 'year']].groupby(['year']).mean().reset_index()
    week_mean.rename(columns={'sales': 'mean_sales'}, inplace=True)

    week_encoding = df[['dayofweek', 'sales', 'year']].groupby(
        ['year', 'dayofweek']).mean().reset_index()
    week_encoding.rename(columns={'sales': 'weekday_avg_sales'}, inplace=True)
    week_encoding = week_encoding.merge(week_mean, how='left', on=['year'])
    week_encoding['weekday_frac_sales'] = week_encoding['weekday_avg_sales'] / \
        week_encoding['mean_sales']
    week_encoding.drop(['mean_sales', 'weekday_avg_sales'], axis=1, inplace=True)

    df = df.merge(week_encoding, how='left', on=['year', 'dayofweek'])
    df = df.sort_values(['unique_id', 'date'])
    df['moving'] = df[['unique_id', 'sales']].groupby('unique_id')['sales']\
        .shift(14).rolling(window=14, min_periods=1).mean().fillna(0)
    df['week_trend'] = df['weekday_frac_sales'] * df['moving']

    # lags
    DAY_PERIODS = [14, 21, 28, 35, 42]
    OFF_PERIODS = [15, 16, 17, 18, 19, 20]
    orders = df['total_orders']
    for shift in DAY_PERIODS:
        grouped = df[['unique_id', 'sales', 'total_orders']].groupby('unique_id')
        sales = grouped['sales'].shift(periods=shift)
        s_orders = grouped['total_orders'].shift(periods=shift)
        df[f'lag_{shift}'] = (orders * sales / s_orders).fillna(0)
        
    numerator = df['weekday_frac_sales'].shift(periods=14)
    for shift in OFF_PERIODS:
        grouped = df[['unique_id', 'sales', 'weekday_frac_sales']].groupby('unique_id')
        sales = grouped['sales'].shift(periods=shift)
        frac = grouped['weekday_frac_sales'].shift(periods=shift)
        df[f'lag_{shift}'] = (numerator * sales / frac).fillna(0)

    df['normed_week_mean'] = df[['lag_14'] + [f'lag_{i}' for i in DAY_PERIODS]].mean(axis=1)
    df['normed_week_median'] = df[['lag_14'] + [f'lag_{i}' for i in DAY_PERIODS]].median(axis=1)

    df = df.drop([f'lag_{i}' for i in OFF_PERIODS], axis=1)
    df = df.drop(['warehouse', 'dayofweek', 'year', 'weekday_frac_sales'], axis=1)

    # date aggregation
    cols = df.columns.to_list()
    for col in ['unique_id', 'date', 'weight']:
        cols.remove(col)
    for col in cols:
        df[col] = df['weight'] * df[col]
    df = df.drop(['unique_id', 'weight'], axis=1)

    cols.remove('total_orders')
    temp = df[['date'] + cols].groupby('date').sum()
    temp['total_orders'] = df[['date', 'total_orders']].groupby('date').mean()
    df = temp.reset_index()
    
    # calendar
    calendar = get_calendar()
    calendar['int_date'] = calendar['date'].astype('int64')/864e11
    calendar['int_date'] = calendar['int_date'] - calendar['int_date'].min()
    calendar['city'] = calendar['warehouse'].apply(lambda x: x.split('_')[0])
    calendar = calendar.loc[calendar['city'] == city, ['holiday_name', 'date', 'int_date']].drop_duplicates()
    
    hcounts = calendar['holiday_name'].value_counts().reset_index()
    hcounts = hcounts.loc[hcounts['count'] > 2]
    holiday_repeats = hcounts['holiday_name'].unique().tolist()
    holiday_repeats.remove('No Holiday')

    calendar.loc[~calendar['holiday_name'].isin(holiday_repeats), 'holiday_name'] = 'Blank'

    df = df.merge(calendar[['date', 'holiday_name', 'int_date']], how='left', on=['date'])
    df['dayofweek'] = df['date'].dt.dayofweek
    df['cos_weekday'] = np.cos(2 * np.pi * df['date'].dt.weekday / 7)
    df['sin_weekday'] = np.sin(2 * np.pi * df['date'].dt.weekday / 7)
    df = df.drop(['date', 'dayofweek'], axis=1)

    return df

class Convex:

    def __init__(self, city: str):
        self.df = get_dataframe(city)
        self.length = len(self.df)

        self.y = cp.Parameter((self.length, 1), value = self.df['sales'].to_numpy().reshape(-1, 1))
        cols = self.df.columns.to_list()
        cols.remove('sales')
        cols.remove('holiday_name')
        cols.remove('int_date')
        self.cols = cols
        self.X = cp.Parameter((self.length, len(cols)), value = self.df[cols].to_numpy())
        self.betas = cp.Variable((self.X.shape[1], 1))

        holidays = self.df.loc[self.df['holiday_name'] != 'Blank', ['holiday_name', 'int_date']]
        
        holiday_names = holidays['holiday_name'].unique().tolist()
        
        self.lr_vars = {holiday : (cp.Variable(nonneg=True), cp.Variable(nonneg=True)) for holiday in holiday_names}

        n = self.length
        vecs = []
        for holiday in holiday_names:
            dates = holidays.loc[holidays['holiday_name'] == holiday, 'int_date'].astype('int64').to_list()
            left, right = self.lr_vars[holiday]
            vec = cp.Parameter((n, 1), value = np.zeros((n, 1)))
            for date in dates:
                l = cp.Parameter((date, 1), value=np.arange(-date, 0).reshape(-1, 1))
                l = cp.exp(left * l)

                r = cp.Parameter((n - date, 1), value=np.arange(0, -(n - date), -1).reshape(-1, 1))
                r = cp.exp(right * r)

                concat = cp.vstack([l, r])
                
                vec += concat

            vecs.append(vec)

        self.hol_X = cp.hstack(vecs)

    def solve(self):
        c = cp.Variable()
        xi = cp.Variable(self.length)

        self.hbetas = cp.Variable((self.hol_X.shape[1], 1), nonneg=True)
        expression = self.y - self.X @ self.betas - self.hol_X @ self.hbetas - c * cp.Parameter(self.y.shape, value=np.ones(self.y.shape))
        constraints = [
            xi >= cp.abs(expression)
        ]

        objective = cp.Minimize(cp.mean(xi))

        problem = cp.Problem(objective, constraints)
        problem.solve()

        # coefficients
        print(c)
        #print(f'{"Constant":30} {c:.4f}')
        for beta, col in zip(self.betas, self.cols):
            print(f'{col:30} {beta:.4f}')
        '''
        for holiday, (left, right) in self.lr_vars.items():
            print(f'---==={holiday}===---')
            print(f'left  {left.value:.4f}')
            print(f'right {right.value:.4f}')
        '''