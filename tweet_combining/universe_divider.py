import glob
import pickle
import logging
import pandas as pd
from typing import Set
from tqdm import tqdm as progress
from collections.abc import Iterable, Generator


def get_duplicates(df_list: Iterable[pd.DataFrame]) -> Set[int]:
    seen = set()
    duplicates = set()
    for df in progress(df_list, total=1163):
        logging.info(f'Starting {df}')
        for row in df.itertuples(index=False, name='Tweet'):
            if row.id in seen:
                logging.info(f'Already seen {row.id}')
                duplicates.add(row.id)
            else:
                seen.add(row.id)

    return duplicates


def get_df_generator() -> Generator[pd.DataFrame, None, None]:
    for file in glob.glob('./out/*.parquet.gzip'):
        yield pd.read_parquet(file)


def main():
    logging.basicConfig(filename='csv_converter.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info('Getting dataframe generator')
    df_generator = get_df_generator()
    duplicates = get_duplicates(df_generator)

    with open('duplicates.pickle', 'wb') as fp:
        pickle.dump(duplicates, fp)

    



        



if __name__ == "__main__":
    main()
