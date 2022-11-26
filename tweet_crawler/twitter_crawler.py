from typing import Callable, Generator, List, Literal, TypeVar
import pandas as pd
import glob


CrawlerType = Literal["test", "full"]
T = TypeVar("T")


class TweetCrawler:

    parquet_count: int
    dfs: List[pd.DataFrame]

    def __init__(self, crawler_type: CrawlerType):
        self.dfs = self.read_dfs(crawler_type)
        self.parquet_count = -1 if crawler_type == "test" else 0  # todo: find these

    def read_dfs(self, crawler_type: CrawlerType) -> List[pd.DataFrame]:
        paths = [file for file in glob.glob("**/*.parquet*") if "likes" not in file]
        return [pd.read_parquet(path) for path in paths]

    def apply_function(self, callable: Callable[[pd.DataFrame], T]) -> Generator[T, None, None]:
        for df in self.dfs:
            yield callable(df)

            
def main():
    pass


if __name__ == "__main__":
    main()
