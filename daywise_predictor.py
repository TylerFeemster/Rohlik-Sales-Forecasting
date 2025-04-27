from datetime import datetime
import numpy as np
import pandas as pd
import gc
from data_utils import add_holidays, process_calendar, process_inventory
import lightgbm as lgb

class FinalSubmitter:

    def __init__(self):

        train = get_train(test_ids_only=True)
        test = get_test()
        inventory = get_inventory()
        calendar = get_calendar()

        train = train.merge(
            inventory, on=['unique_id', 'warehouse'], how='left')
        train = train.merge(calendar, on=['date', 'warehouse'], how='left')

        test = test.merge(inventory, on=['unique_id', 'warehouse'], how='left')
        test = test.merge(calendar, on=['date', 'warehouse'], how='left')

        weights = pd.read_csv('./data/test_weights.csv')
        train = train.merge(weights, on=['unique_id'], how='left')
        test = test.merge(weights, on=['unique_id'], how='left')

        train, test = double_fe(train, test)
        self.train, self.test = self.__preprocess(train, test)

        self.model = None
        return

    def __preprocess(self, train: pd.DataFrame, test: pd.DataFrame):
        drop_cols = ['mean_hol_sales']

        train.drop(drop_cols, axis=1, inplace=True)
        test.drop(drop_cols, axis=1, inplace=True)

        categorical_columns = ['unique_id'] + \
            list(train.select_dtypes("object").columns)
        for col in categorical_columns:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')

        return train, test

    def build_model(self):

        train_data = self.train
        test_data = self.test

        X_train = train_data.drop(['sales', 'date', 'weight'], axis=1)
        y_train = train_data['sales']
        train_weights = train_data['weight']

        X_test = test_data.drop(['date', 'weight'], axis=1)

        categorical_cols = ['unique_id'] + \
            list(self.train.select_dtypes("object").columns)

        callbacks = [lgb.log_evaluation(period=100)]

        params = {
            'learning_rate': 0.04,
            'num_leaves': 80,
            'subsample': 0.7,
            'colsample_bytree': 0.8528497905459008,
            'reg_lambda': 0.3151110021900479,
            'num_boost_round': 5000,
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
        }

        train_dataset = lgb.Dataset(X_train, label=y_train,
                                    categorical_feature=categorical_cols,
                                    weight=train_weights)

        self.model = lgb.train(params,
                               train_dataset,
                               valid_sets=[train_dataset],
                               valid_names=['train'],
                               callbacks=callbacks)

        y_pred = self.model.predict(
            X_test.loc[self.test.index],
            num_iteration=self.model.best_iteration)
        sub = self.test.copy()
        sub['sales'] = y_pred
        sub[['unique_id', 'date', 'sales', 'weight']].to_csv(
            "base_test_forecasts.csv", index=False)
        return

# --------------------------
# Initializing Data
# --------------------------


def __sanitize_train(df: pd.DataFrame):
    df = df.fillna(0)
    return df


def get_train(test_ids_only=False):
    df = pd.read_csv('./data/sales_train.csv')

    if test_ids_only:
        test_ids = pd.read_csv(
            './data/sales_test.csv')['unique_id'].unique().tolist()
        df = df[df['unique_id'].isin(test_ids)]

    df = __sanitize_train(df)
    df = feature_engineering(df)

    return df


def get_test():
    df = pd.read_csv('./data/sales_test.csv')
    df = feature_engineering(df)
    return df


def get_inventory():
    df = pd.read_csv('./data/inventory.csv')
    df = process_inventory(df)
    return df


def get_calendar():
    df = pd.read_csv('./data/calendar.csv')
    df = add_holidays(df)
    df = process_calendar(df)
    return df

# ---------------------------
# Data Preprocessing
# ---------------------------


def __date_features(df: pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day.clip(upper=30)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear

    df['weekend'] = (df['dayofweek'] > 4).astype(int)

    return df


def __price_features(df: pd.DataFrame):
    discounts = [f'type_{i}_discount' for i in range(7)]
    df['top_discount'] = df[discounts].max(axis=1)
    df['discount_price'] = df['sell_price_main'] * (1 - df['top_discount'])
    df['discount_amount'] = df['sell_price_main'] * df['top_discount']
    df['total_pred_sales'] = df['total_orders'] * df['sell_price_main']
    df['total_pred_discount_sales'] = df['total_orders'] * df['discount_price']
    df['order_price_ratio'] = df['total_orders'] / df['sell_price_main']
    df['order_discount_ratio'] = df['total_orders'] / df['discount_price']

    mean_prices = df.groupby(df['unique_id'])['sell_price_main'].mean()
    std_prices = df.groupby(df['unique_id'])['sell_price_main'].std()
    df['price_scaled'] = np.where(df['unique_id'].map(std_prices) == 0, 0,
                                  (df['sell_price_main'] - df['unique_id'].map(mean_prices))/df['unique_id'].map(std_prices))
    df['days_since_2020'] = (
        df['date'] - pd.to_datetime('2020-01-01')).dt.days.astype('int')
    df['price_detrended'] = df['price_scaled'] - \
        df.groupby(['days_since_2020', 'warehouse'])[
        'price_scaled'].transform('mean')
    df = df.drop(['price_scaled', 'days_since_2020'], axis=1)

    return df


def feature_engineering(df: pd.DataFrame):

    discounts = [f'type_{i}_discount' for i in range(7)]
    for discount in discounts:
        df.loc[df[discount] < 0, discount] = 0

    df = __date_features(df)
    df = __price_features(df)

    df['city'] = df['warehouse'].apply(lambda x: x.split('_')[0])
    country = {
        'Budapest_1': 'Hungary',
        'Prague_2': 'Czechia',
        'Brno_1': 'Czechia',
        'Prague_1': 'Czechia',
        'Prague_3': 'Czechia',
        'Munich_1': 'Germany',
        'Frankfurt_1': 'Germany'
    }
    df['country'] = df['warehouse'].apply(lambda x: country[x])

    return df


def __relative_price(df: pd.DataFrame):
    cols = [f'L{i}' for i in range(1, 5)] + ['kind']
    for col in cols:
        temp = df[['date', 'warehouse', 'discount_price', col]]
        stats = temp.groupby(
            ['warehouse', col, 'date'], observed=True).min().reset_index()
        stats['min'] = stats['discount_price']  # name fix
        stats = stats.drop('discount_price', axis=1)

        mean_df = temp.groupby(['warehouse', col, 'date'],
                               observed=True).mean().reset_index()
        stats['mean'] = mean_df['discount_price']
        gc.collect()

        merged = temp.merge(stats, how='left',
                            on=['warehouse', col, 'date'])

        # for category, date, and warehouse, asks if least price (takes into account current discounts)
        df[f'relative_price_{col}'] = ((merged['discount_price'] - merged['min'])
                                       / (merged['mean'] - merged['min'])).fillna(1)
        
    if 'availability' not in df.columns.to_list():
        df['availability'] = 1

    for col in cols:
        temp = df[['date', 'warehouse', 'availability', col]]
        temp['avg_avail'] = temp.groupby(['warehouse', col, 'date'], observed=True)\
            .transform('mean')['availability']
        temp[f'relative_avail_{col}'] = temp['availability'] - temp['avg_avail']

        df = df.merge(temp[[f'relative_avail_{col}', 'date', 'warehouse', col]],
                      how='left', on=['date', 'warehouse', col])
    
    df = df.drop('availability', axis=1)

    return df


def __modify_holiday(df: pd.DataFrame):
    # weekday holidays are more anamolous
    df['holiday'] = df['holiday'] + df['holiday'] * (1 - df['weekend'])
    return df


def __count_encoding(train: pd.DataFrame, test: pd.DataFrame):
    cols = ['unique_id', 'date', 'holiday_name',
            'warehouse', 'sell_price_main']
    full = pd.concat([train[cols], test[cols]]).sort_values(
        by=['unique_id', 'date'])
    full['cum_count'] = full[['unique_id', 'date']
                             ].groupby('unique_id').cumcount()

    unique_prices = full[['unique_id', 'sell_price_main']]\
        .groupby('unique_id').nunique().reset_index()
    unique_prices.rename(
        columns={'sell_price_main': 'price_counts'}, inplace=True)
    full = full.merge(unique_prices, how='left', on='unique_id')

    cols.remove('date')
    cols.remove('sell_price_main')

    for col in cols:
        counts = full[col].value_counts().reset_index()
        counts.rename(columns={'count': f'{col}_count'}, inplace=True)
        full = full.merge(counts, how='left', on=col)

    new_cols = [f'{col}_count' for col in cols] + ['price_counts', 'cum_count']
    full = full[['unique_id', 'date'] + new_cols]
    gc.collect()

    train_part = full.iloc[:len(train)].reset_index(drop=True)
    test_part = full.iloc[len(train):].reset_index(drop=True)

    train = train.merge(train_part, how='left', on=['unique_id', 'date'])
    test = test.merge(test_part, how='left', on=['unique_id', 'date'])

    return train, test


def __target_encoding(train: pd.DataFrame, test: pd.DataFrame):
    t_cols = ['sales', 'date', 'unique_id']
    test_copy = test.copy()
    test_copy['sales'] = 0
    df = pd.concat([train[t_cols], test_copy[t_cols]])
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    # Calculate monthly averages per product
    monthly_avg = (
        df.groupby(['unique_id', 'month'])['sales']
        .mean()
        .groupby('unique_id')
        .shift(1)
        .reset_index()
        .rename(columns={'sales': 'prev_month_avg'})
    )
    df = df.merge(monthly_avg, on=['unique_id', 'month'], how='left').fillna(0)
    df.drop(columns=['month', 'sales'], inplace=True)

    train = train.merge(df, how='left', on=['unique_id', 'date'])
    test = test.merge(df, how='left', on=['unique_id', 'date'])
    gc.collect()

    # holiday target encoding
    by_hol_id = train[['unique_id', 'holiday_name', 'sales']].groupby(
        ['unique_id', 'holiday_name'])

    mean_hol_id = by_hol_id.mean().fillna(0).reset_index()
    mean_hol_id.rename(columns={'sales': 'mean_hol_sales'}, inplace=True)
    std_hol_id = by_hol_id.std().fillna(0).reset_index()
    std_hol_id.rename(columns={'sales': 'std_hol_sales'}, inplace=True)
    hol_stats = pd.concat(
        [mean_hol_id, std_hol_id.loc[:, 'std_hol_sales']], axis=1)
    hol_stats['log_cv'] = (hol_stats['std_hol_sales'] / hol_stats['mean_hol_sales']
                           ).fillna(0).apply(lambda x: np.log(x + 1e-3))

    keep_cols = ['mean_hol_sales', 'std_hol_sales',
                 'unique_id', 'holiday_name', 'log_cv']
    train = train.merge(hol_stats[keep_cols], how='left', on=[
                        'unique_id', 'holiday_name'])
    test = test.merge(hol_stats[keep_cols], how='left', on=[
                      'unique_id', 'holiday_name'])

    gc.collect()

    train['weighted_std'] = train['std_hol_sales'] * train['weight']
    test['weighted_std'] = test['std_hol_sales'] * test['weight']

    gc.collect()

    # robust week encoding
    week_encoding = train[['dayofweek', 'sales', 'city', 'year']].groupby(
        ['year', 'dayofweek', 'city']).mean().reset_index()
    week_mean = train[['sales', 'city', 'year']].groupby(
        ['year', 'city']).mean().reset_index()
    week_mean.rename(columns={'sales': 'mean_sales'}, inplace=True)
    week_encoding.rename(columns={'sales': 'weekday_avg_sales'}, inplace=True)
    week_encoding = week_encoding.merge(
        week_mean, how='left', on=['city', 'year'])
    week_encoding['weekday_frac_sales'] = week_encoding['weekday_avg_sales'] / \
        week_encoding['mean_sales']

    week_encoding.drop('mean_sales', axis=1, inplace=True)

    train = train.merge(week_encoding, how='left', on=[
                        'year', 'dayofweek', 'city'])
    test = test.merge(week_encoding, how='left', on=[
                      'year', 'dayofweek', 'city'])

    train['week_trend'] = train['weekday_frac_sales'] * train['prev_month_avg']
    test['week_trend'] = test['weekday_frac_sales'] * test['prev_month_avg']

    return train, test


def __trend(train: pd.DataFrame, test: pd.DataFrame):

    all_cols = ['product_sales_14', 'product_sales_17', 'product_sales_21', 'product_sales_28',
                'product_sales_35', 'product_sales_42', 'weekday_avg_sales', 'week_trend']
    # all
    coef_df = pd.read_csv('./trend_coefs_14.csv')
    train = train.merge(coef_df, how='left', on='unique_id')
    test = test.merge(coef_df, how='left', on='unique_id')
    train['trend'] = train['intercept']
    test['trend'] = test['intercept']
    for col in all_cols:
        train['trend'] += train[col].fillna(0) * train[f'coef_{col}']
        test['trend'] += test[col].fillna(0) * test[f'coef_{col}']
    drop_cols = ['intercept'] + [f'coef_{col}' for col in all_cols]
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)

    gc.collect()

    # new
    nec_cols = ['product_sales_14', 'product_sales_21', 'product_sales_28', 'product_sales_35',
                'weekday_avg_sales', 'week_trend', 'week_moving_trend', 'moving', 'normed_week_mean',
                'normed_week_median'] + [f'lag_{i}' for i in [14, 21, 28, 35]]
    coef_df = pd.read_csv('./trend_coefs_new.csv')
    train = train.merge(coef_df, how='left', on='unique_id')
    test = test.merge(coef_df, how='left', on='unique_id')
    train['trend_new'] = train['intercept']
    test['trend_new'] = test['intercept']
    for col in nec_cols:
        train['trend'] += train[col].fillna(0) * train[f'coef_{col}']
        test['trend'] += test[col].fillna(0) * test[f'coef_{col}']
    drop_cols = ['intercept'] + [f'coef_{col}' for col in nec_cols]
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)

    return train, test


def __spectral_embedding(train: pd.DataFrame, test: pd.DataFrame):
    embed = pd.read_csv('id_embeddings.csv')

    train = train.merge(embed, how='left', on='unique_id')
    test = test.merge(embed, how='left', on='unique_id')

    return train, test


def double_fe(train: pd.DataFrame, test: pd.DataFrame):

    train, test = __target_encoding(train, test)
    train, test = __count_encoding(train, test)

    # df feature engineering
    week_mean = train[['sales', 'year', 'city']].groupby(
        ['year', 'city']).mean().reset_index()
    week_mean.rename(columns={'sales': 'mean_sales'}, inplace=True)

    week_encoding = train[['dayofweek', 'sales', 'year', 'city']].groupby(
        ['year', 'dayofweek', 'city']).mean().reset_index()
    week_encoding.rename(columns={'sales': 'weekday_avg_sales'}, inplace=True)
    week_encoding = week_encoding.merge(
        week_mean, how='left', on=['year', 'city'])
    week_encoding['weekday_frac_sales'] = week_encoding['weekday_avg_sales'] / \
        week_encoding['mean_sales']
    week_encoding.drop(['mean_sales', 'weekday_avg_sales'],
                       axis=1, inplace=True)

    # lags
    PERIODS = [14, 17, 21, 28, 35, 42, 63, 91, 182, 364, 728]
    train_start = datetime.strptime(
        '08-01-2020', '%m-%d-%Y')
    train_end = datetime.strptime(
        '06-02-2024', '%m-%d-%Y')

    train_slice = train[['unique_id', 'date', 'sales',
                         'year', 'dayofweek', 'total_orders', 'city']]
    test_slice = test[['unique_id', 'date', 'year',
                       'dayofweek', 'total_orders', 'city']].copy()
    test_slice['sales'] = 0

    combo = pd.concat([train_slice, test_slice],
                      ignore_index=True).sort_values(['unique_id', 'date'])
    for shift in PERIODS:
        combo[f'product_sales_{shift}'] = (combo.groupby('unique_id', observed=True)['sales']
                                           .transform(lambda x: x.shift(shift).fillna(0))
                                           )

    df = combo.merge(week_encoding, how='left', on=[
                     'year', 'dayofweek', 'city'])
    df = df.sort_values(['unique_id', 'date'])
    df['moving'] = (df.groupby('unique_id')['sales']
                    .transform(lambda x: x.shift(14).rolling(window=14, min_periods=1).mean().fillna(0))
                    )
    df['week_moving_trend'] = df['weekday_frac_sales'] * df['moving']

    # lags
    DAY_PERIODS = [14, 21, 28, 35, 42]
    OFF_PERIODS = [15, 16, 17, 18, 19, 20]
    orders = df['total_orders']
    for shift in DAY_PERIODS:
        grouped = df[['unique_id', 'sales', 'total_orders']
                     ].groupby('unique_id')
        sales = grouped['sales'].transform(lambda x: x.shift(shift))
        s_orders = grouped['total_orders'].transform(lambda x: x.shift(shift))
        df[f'lag_{shift}'] = (orders * sales / s_orders).fillna(0)

    numerator = df['weekday_frac_sales'].shift(periods=14)
    for shift in OFF_PERIODS:
        grouped = df[['unique_id', 'sales', 'weekday_frac_sales']
                     ].groupby('unique_id')
        sales = grouped['sales'].transform(lambda x: x.shift(shift))
        frac = grouped['weekday_frac_sales'].transform(
            lambda x: x.shift(shift))
        df[f'lag_{shift}'] = (numerator * sales / frac).fillna(0)

    df['normed_week_mean'] = df[['lag_14'] +
                                [f'lag_{i}' for i in DAY_PERIODS]].mean(axis=1)
    df['normed_week_median'] = df[['lag_14'] +
                                  [f'lag_{i}' for i in DAY_PERIODS]].median(axis=1)

    df = df.drop([f'lag_{i}' for i in OFF_PERIODS], axis=1)
    df = df.drop(['dayofweek', 'year', 'weekday_frac_sales',
                 'total_orders', 'city'], axis=1)

    train_info = df.loc[(df['date'] >= train_start)
                        & (df['date'] <= train_end)].drop('sales', axis=1)
    test_info = df.loc[df['date'] > train_end].drop('sales', axis=1)

    train_info['date'] = pd.to_datetime(train_info['date'])
    test_info['date'] = pd.to_datetime(test_info['date'])

    train = train.merge(train_info, how='left', on=['unique_id', 'date'])
    test = test.merge(test_info, how='left', on=['unique_id', 'date'])

    gc.collect()

    train = __relative_price(train)
    test = __relative_price(test)

    train = __modify_holiday(train)
    test = __modify_holiday(test)

    # spectral embedding
    train, test = __spectral_embedding(train, test)

    train, test = __trend(train, test)

    return train, test
