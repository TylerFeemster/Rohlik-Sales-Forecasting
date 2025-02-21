from sklearn.manifold import SpectralEmbedding
import pandas as pd
import numpy as np


class TargetEncoder:

    def __init__(self, group_col, target_col='sales'):
        self.group = group_col
        self.target = target_col

    def fit(self, df):
        slice = df[[self.group, 'warehouse', self.target]]
        self.stats = slice.groupby([self.group, 'warehouse']).agg(
            n=(self.target, "count"),
            mu_mle=(self.target, "mean"),
            sig2_mle=(self.target, "var"))
        self.stats['mu_prior'] = df[self.target].mean()
        return

    def transform(self, df, prior_precision=5):

        post_precision = self.stats.n / self.stats.sig2_mle
        precision = prior_precision + post_precision

        numer = prior_precision * self.stats.mu_prior \
            + post_precision * self.stats.mu_mle
        denom = precision

        new_vals = ((numer / denom).reset_index())\
            .rename(columns={0: f'GTE_{self.group}'})

        df = df.merge(new_vals, how='left', on=[self.group, 'warehouse'])
        return df

    def fit_transform(self, df, *args, **kwargs):
        self.fit(df)
        return self.transform(df, *args, **kwargs)


class HierarchicalTargetEncoder:

    def __init__(self, parent_col, child_col, target_col='sales'):
        self.parent = parent_col
        self.child = child_col
        self.target = target_col

    def __get_priors(self, df):
        return df[[self.parent, 'warehouse', self.target]].groupby([self.parent, 'warehouse']).mean()\
            .reset_index().rename(columns={self.target: 'mu_prior'})

    def fit(self, df):
        slice = df[[self.parent, self.child, 'warehouse', self.target]]
        slice = slice.merge(self.__get_priors(
            slice), how='left', on=[self.parent, 'warehouse'])
        self.stats = slice.groupby([self.child, 'warehouse']).agg(
            n=(self.target, "count"),
            mu_mle=(self.target, "mean"),
            sig2_mle=(self.target, "var"),
            mu_prior=('mu_prior', "mean"))
        return

    def transform(self, df, prior_precision=5):

        post_precision = self.stats.n / self.stats.sig2_mle
        precision = prior_precision + post_precision

        numer = prior_precision * self.stats.mu_prior \
            + post_precision * self.stats.mu_mle
        denom = precision

        new_vals = ((numer / denom).reset_index()
                    ).rename(columns={0: f'HTE_{self.child}/{self.parent}'})

        df = df.merge(new_vals, how='left', on=[self.child, 'warehouse'])
        return df

    def fit_transform(self, df, *args, **kwargs):
        self.fit(df)
        return self.transform(df, *args, **kwargs)


def monthly_target_encode(train: pd.DataFrame, test: pd.DataFrame, group_cols=[],
                          target_col='sales', prior_precision=5):
    train = train.copy()
    test = test.copy()
    train['mnth'] = train['date'].dt.to_period('M')
    test['mnth'] = test['date'].dt.to_period('M')

    trn_slice = train[group_cols + ['mnth', 'city', target_col]]
    priors = trn_slice[[target_col, 'mnth', 'city']].groupby(['mnth', 'city']).mean()\
        .reset_index().rename(columns={target_col: 'mu_prior'})
    priors['mu_prior'] = priors['mu_prior'].fillna(train[target_col].mean())

    for group in group_cols:
        stats = trn_slice[[group, target_col, 'mnth', 'city']]\
            .groupby([group, 'mnth', 'city']).agg(
            n=(target_col, "count"),
            mu_mle=(target_col, "mean"),
            sig2_mle=(target_col, "var")).reset_index()

        stats['sig2_mle'] = stats['sig2_mle'].fillna(1).replace(0, 1)

        stats = stats.merge(priors, how='left', on=['mnth', 'city'])

        stats['post'] = stats['n'] / stats['sig2_mle']
        stats['precision'] = prior_precision + stats['post']

        stats['numer'] = prior_precision * stats['mu_prior'] \
            + stats['post'] * stats['mu_mle']

        stats[f'MCGTE_{group}'] = stats['numer'] / stats['precision']

        stats['mnth'] = stats['mnth'] + 1
        final = stats[[f'MCGTE_{group}', group, 'mnth', 'city']]

        train = train.merge(final, how='left', on=[
                            'mnth', group, 'city']).fillna(0)
        test = test.merge(final, how='left', on=[
                          'mnth', group, 'city']).fillna(0)

    train.drop('mnth', axis=1, inplace=True)
    test.drop('mnth', axis=1, inplace=True)

    return train, test


def spectral_encode():
    n_ids = 5432
    dates = 1402

    df = pd.read_csv('./data/sales_train.csv')[['unique_id', 'date', 'sales']]
    df.dropna(inplace=True)
    df['date'] = (pd.to_datetime(df['date']).astype(
        'int64')/864e11).astype('int64')
    dmin = df['date'].min()
    df['date'] -= dmin

    means = df[['unique_id', 'sales']].groupby('unique_id')\
        .mean().reset_index().rename(columns={'sales': 'mean'})
    stds = df[['unique_id', 'sales']].groupby('unique_id')\
        .std().fillna(1).reset_index().rename(columns={'sales': 'std'})
    df = df.merge(means, how='left', on='unique_id')\
        .merge(stds, how='left', on='unique_id')
    df['normed_sales'] = (df['sales'] - df['mean']) / df['std']

    matr1 = np.zeros((n_ids, dates))
    matr2 = np.zeros((n_ids, dates))
    for _, row in df.iterrows():
        matr1[int(row.unique_id), int(row.date)] = row.sales
        matr2[int(row.unique_id), int(row.date)] = row.normed_sales
    matr1 = np.nan_to_num(matr1, nan=0.0, posinf=0, neginf=0)
    matr2 = np.nan_to_num(matr2, nan=0.0, posinf=0, neginf=0)

    emb = SpectralEmbedding(random_state=2, n_jobs=-1)
    embedded = 1000 * emb.fit_transform(matr1)

    embed1 = pd.DataFrame(embedded[:, 0]).reset_index()\
        .rename(
        columns={
            'index': 'unique_id',
            0: 'embed_0'
        }
    )

    emb = SpectralEmbedding(random_state=2, n_jobs=-1)
    embedded = 200 * emb.fit_transform(matr2)
    embed2 = pd.DataFrame(embedded).reset_index()\
        .rename(
        columns={
            'index': 'unique_id',
            0: 'embed_1',
            1: 'embed_2'
        }
    )

    embed = embed1.merge(embed2, how='left', on='unique_id')
    embed.to_csv('id_embeddings.csv', index=False)

    return