import glob
import pickle
import pandas as pd
from tqdm import tqdm as progress
from typing import Callable, Generator, List, Literal, Set, Tuple, TypeVar


CrawlerType = Literal['test', 'harvard', 'likes', 'left']
T = TypeVar('T')


DATA_PATH = '/home/ec2-user/congressional_tweets'


class TweetCrawler:

    parquet_count: int
    dfs: List[pd.DataFrame]

    def __init__(self, crawler_type: CrawlerType):
        self.dfs = self.read_dfs(crawler_type)
        if crawler_type == 'test':
            self.parquet_count = 1
        elif crawler_type == 'full':
            self.parquet_count = 11
        else:
            self.parquet_count = -1

    def read_dfs(self, crawler_type: CrawlerType) -> List[pd.DataFrame]:
        if crawler_type == 'harvard':
            paths = [file for file in glob.glob(f'{DATA_PATH}/full_harvard/*parquet*', recursive=True) if 'likes' not in file]
        elif crawler_type == 'test':
            paths = [f'{DATA_PATH}/senators_115.parquet.gzip']
        elif crawler_type =='left':
            paths = [file for file in glob.glob(f'{DATA_PATH}/**/*parquet*', recursive=True) if 'likes' in file or 'retweets' in file]
        else:
            paths = [file for file in glob.glob(f'{DATA_PATH}/**/*parquet*', recursive=True) if 'likes' in file]
        return [pd.read_parquet(path) for path in progress(paths, desc='Reading dataframes')]

    def apply_function(self, callable: Callable[[pd.DataFrame], T]) -> Generator[T, None, None]:
        for df in self.dfs:
            yield callable(df)


def get_ids(df: pd.DataFrame) -> Set[int]:
    return set(df['id'].to_list())


def get_count(df: pd.DataFrame) -> int:
    return len(df['id'])


def get_count_by_document(df: pd.DataFrame) -> pd.DataFrame:
    df['day'] = pd.DatetimeIndex(df.created_at).normalize()
    return df.groupby(['day']).count()

def get_user_ids(df: pd.DataFrame) -> Set[int]:
    return set(df['author_id'].to_list())

def get_retweets(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['tweet_type'] == 'retweet']


def test():
    left = TweetCrawler('left')
    retweets = left.apply_function(get_user_ids)

    for i in retweets:
        print(i)


def main():
    left_crawler = TweetCrawler('harvard')
    retweets = left_crawler.apply_function(get_retweets)
    all_op_tweets = set()

    for retweet_df in progress(retweets):
        original_tweets = set(retweet_df['original_tweet_id'].to_list())
        all_op_tweets.update(original_tweets)
        
    with open('originals.txt', 'w') as fp:
        for user_id in all_op_tweets:
            fp.write(f'{user_id:.20f}')
            fp.write('\n')



def likes():
    likes_crawler = TweetCrawler('likes')
    author_ids = likes_crawler.apply_function(get_user_ids)
    total_ids = set()
    for id_set in author_ids:
        total_ids.update(id_set)

    with open('liked_user_ids.txt', 'w') as fp:
        for user_id in total_ids:
            fp.write(str(user_id))
            fp.write('\n')

    # likes_gb = likes_crawler.apply_function(get_count)
    # count = 0
    # for a in likes_gb:
    #     count += a
    # print(count)
    # counts = pd.concat(list(likes_gb), axis=1).sum(axis=1)
    # with open('counts.pickle', 'wb') as fp:
    #     pickle.dump(counts, fp)



if __name__ == "__main__":
    likes()
