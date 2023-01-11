import os 
from tqdm import tqdm as progress
import pandas as pd
import shutil


def main():
    print('Reading directories...')
    round_1 = set(os.listdir('./round_1'))
    round_2 = set(os.listdir('./round_2'))

    only_in_one = round_1.symmetric_difference(round_2)
    in_both = round_1.intersection(round_2)

    for user_likes in progress(only_in_one):
        try:
            shutil.copy2(f'./round_1/{user_likes}', './out/')
        except FileNotFoundError:
            shutil.copy2(f'./round_2/{user_likes}', './out/')

    for user_likes in progress(in_both):
        df1 = pd.read_parquet(f'./round_1/{user_likes}')
        df2 = pd.read_parquet(f'./round_2/{user_likes}')
        df3 = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        df3.to_parquet(f'./out/{user_likes}',
                       index=False,
                       compression='gzip')



if __name__ == "__main__":
    main()
