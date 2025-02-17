from sklearn.manifold import SpectralEmbedding
import pandas as pd
import numpy as np

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
