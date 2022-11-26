import glob
import pandas as pd
from typing import Callable, Generator, List, Literal, TypeVar


CrawlerType = Literal["test", "full"]
T = TypeVar("T")


DATA_PATH = '/home/ec2-user/congressional_tweets'


class TweetCrawler:

    parquet_count: int
    dfs: List[pd.DataFrame]

    def __init__(self, crawler_type: CrawlerType):
        self.dfs = self.read_dfs(crawler_type)
        self.parquet_count = -1 if crawler_type == "test" else 0  # todo: find these

    def read_dfs(self, crawler_type: CrawlerType) -> List[pd.DataFrame]:
        if crawler_type == 'full':
            paths = [file for file in glob.glob(f'{DATA_PATH}/*parquet*') if 'likes' not in file]
        else:
            paths = [f'{DATA_PATH}/senators_115.parquet.gzip']
        return [pd.read_parquet(path) for path in paths]

    def apply_function(self, callable: Callable[[pd.DataFrame], T]) -> Generator[T, None, None]:
        for df in self.dfs:
            yield callable(df)


def main():
    test_crawler = TweetCrawler('full')
    print_generator = test_crawler.apply_function(print)
    for d in print_generator:
        d


if __name__ == "__main__":
    main()
