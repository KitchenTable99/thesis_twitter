import pandas as pd


def add_to_hdf(df, store):
    store.append("df", df, datacolumns=True)


def get_dataframes():
    return []


def main():
    store = pd.HDFStore("all_congressional_tweets.h5")
    df_list = get_dataframes()

    for df in df_list:
        add_to_hdf(df, store)


if __name__ == "__main__":
    main()
