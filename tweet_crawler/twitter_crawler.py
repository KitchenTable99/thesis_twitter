import glob
import pickle
import pandas as pd
from tqdm import tqdm as progress
from typing import Callable, Generator, List, Literal, Set, TypeVar


CrawlerType = Literal["test", "full"]
T = TypeVar("T")


DATA_PATH = '/home/ec2-user/congressional_tweets'


class TweetCrawler:

    parquet_count: int
    dfs: List[pd.DataFrame]

    def __init__(self, crawler_type: CrawlerType):
        self.dfs = self.read_dfs(crawler_type)
        self.parquet_count = 1 if crawler_type == "test" else 11

    def read_dfs(self, crawler_type: CrawlerType) -> List[pd.DataFrame]:
        if crawler_type == 'full':
            paths = [file for file in glob.glob(f'{DATA_PATH}/*parquet*') if 'likes' not in file]
        else:
            paths = [f'{DATA_PATH}/senators_115.parquet.gzip']
        return [pd.read_parquet(path) for path in progress(paths, desc='Reading dataframes')]

    def apply_function(self, callable: Callable[[pd.DataFrame], T]) -> Generator[T, None, None]:
        for df in self.dfs:
            yield callable(df)


def get_ids(df: pd.DataFrame) -> Set[int]:
    return set(df['id'].to_list())


def main():
    full_crawler = TweetCrawler('full')
    total_set = set()
    id_sets = full_crawler.apply_function(get_ids)
    for _set in id_sets:
        total_set.update(_set)

    with open('downloaded_tweet_ids.pickle', 'wb') as fp:
        pickle.dump(total_set, fp)


if __name__ == "__main__":
    main()
