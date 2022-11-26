import pandas as pd
from tqdm import tqdm as progress
import numpy as np
import glob


def add_to_hdf(df, store):
    list_df = np.array_split(df, 1_000)
    for df in progress(list_df, leave=False):
        store.append("df", df, data_columns=True,
                     min_itemsize={
                         'text': 300,
                         'hashtags': 1_000,
                         'mentions': 10_000,
                         'urls': 4_000,
                         'author_username': 50,
                         'cashtags': 300,
                     })


def get_dataframes():
    paths = [file for file in glob.glob('**/*.parquet*') if 'likes' not in file]
    return [pd.read_parquet(path) for path in paths]


def main():
    store = pd.HDFStore("all_congressional_tweets.h5")
    df_list = get_dataframes()

    for df in progress(df_list):
        add_to_hdf(df, store)
    store.close()


if __name__ == "__main__":
    main()
