import os
import glob
import pickle
import logging
import pandas as pd
from tqdm import tqdm as progress
from twarc_csv import CSVConverter
from congressional_representatives import Representative, RepresentativeUniverse, Term


def convert_to_df(file):
    file_name = file.split('/')[-1].split('.')[0]
    logging.info(f'Converting {file_name}')
    if 'test' not in file_name:
        with open(file, 'r') as infile:
            with open(f'./temp/{file_name}.csv', 'w') as outfile:
                converter = CSVConverter(infile=infile, outfile=outfile)
                converter.process()



def determine_tweet_type(row):
    replied_id = row['referenced_tweets.replied_to.id']
    retweet_id = row['referenced_tweets.retweeted.id']
    quote_id = row['referenced_tweets.quoted.id']

    if not pd.isna(replied_id):
        return 'reply'
    elif not pd.isna(retweet_id):
        return 'retweet'
    elif not pd.isna(quote_id):
        return 'qrt'
    else:
        return 'tweet'


def remove_out_of_office_tweets(df, rep: Representative):
    boundaries = rep.get_total_term_boundaries()
    accepted_likes = []
    for boundary in boundaries:
        first_date = str(boundary[0])
        last_date = str(boundary[1])
        logging.debug(f'{first_date = }, {last_date = }')
        in_session_likes = df[(df['created_at'] >= first_date) & (df['created_at'] < last_date)]
        accepted_likes.append(in_session_likes)

    new_df = pd.concat(accepted_likes)
    logging.info(f'Removed {len(df) - len(new_df)} likes')
    logging.debug(f"Oldest tweet: {new_df['created_at'].min()}")
    logging.debug(f"Newest tweet: {new_df['created_at'].max()}")
    return new_df


def process_dataframe(df, file_name, rep_universe: RepresentativeUniverse, write_name: str = None):
        # process columns
        df['tweet_type'] = df.apply(determine_tweet_type, axis=1)
        df['congressional_like'] = True if 'likes' in file_name else False

        # drop unneeded columns
        df = df.drop(columns=['referenced_tweets.replied_to.id', 'referenced_tweets.retweeted.id', 'referenced_tweets.quoted.id'])

        # drop unneeded rows
        if 'likes' in file_name:
            logging.info('Dropping out of bounds likes')
            rep_id = int(file_name.split('_')[0])
            rep = rep_universe.get_rep(rep_id)
            if not rep:
                logging.critical(f'{rep_id = } fucked up somehow')
                raise NotImplemented
            df = remove_out_of_office_tweets(df, rep)

        # rename columns
        df = df.rename(columns={'public_metrics.like_count': 'like_count',
                                'public_metrics.quote_count': 'qrt_count',
                                'public_metrics.reply_count': 'reply_count',
                                'public_metrics.retweet_count': 'retweet_count',
                                'entities.cashtags': 'cashtags',
                                'entities.hashtags': 'hashtags',
                                'entities.mentions': 'mentions',
                                'entities.urls': 'urls',
                                'author.username': 'author_username',
                                'author.name': 'author_name'})

        # set proper column types
        df = df.astype({'tweet_type': 'category',
                        'created_at': 'datetime64[ns]',
                        'lang': 'category',
                        'cashtags': 'object'})

        # write
        to_write = write_name if write_name else file_name
        df.to_parquet(f'./out/{to_write}.parquet.gzip',
                      index=False,
                      compression='gzip')


def create_rep_universe() -> RepresentativeUniverse:
    with open('rep_universe.pickle', 'rb') as fp:
        return pickle.load(fp)


def process_df(file, rep_universe, write_name=None):
    file_name = file.split('/')[-1].split('.')[0]
    logging.info(f'Converting {file_name}')

    file_size = os.stat(file).st_size
    if file_size < 7e9:
        df: pd.DataFrame = pd.read_csv(file,
                                       low_memory=False,
                                       usecols=['id', 'author_id', 'created_at', 'text', 'lang',
                                                'public_metrics.like_count', 'public_metrics.quote_count',
                                                'public_metrics.reply_count', 'public_metrics.retweet_count',
                                                'possibly_sensitive', 'entities.cashtags', 'entities.hashtags',
                                                'entities.mentions', 'entities.urls', 'author.username', 'author.name',
                                                'referenced_tweets.replied_to.id', 'referenced_tweets.retweeted.id',
                                                'referenced_tweets.quoted.id'])
        process_dataframe(df, file_name, rep_universe, write_name=write_name)
    else:
        logging.info('File too big. Breaking into chunks.')
        with pd.read_csv(file,
                         low_memory=False,
                         usecols=['id', 'author_id', 'created_at', 'text', 'lang',
                                  'public_metrics.like_count', 'public_metrics.quote_count',
                                  'public_metrics.reply_count', 'public_metrics.retweet_count',
                                  'possibly_sensitive', 'entities.cashtags', 'entities.hashtags',
                                  'entities.mentions', 'entities.urls', 'author.username', 'author.name',
                                  'referenced_tweets.replied_to.id', 'referenced_tweets.retweeted.id',
                                  'referenced_tweets.quoted.id'],
                         chunksize=453_241) as reader:  # chunksize calculated by estimating number of lines that will take up 3GB
            for idx, chunk in progress(enumerate(reader), leave=False):
                logging.info(f'Converting chunk #{idx}')
                process_dataframe(chunk, file_name=file_name + str(idx), rep_universe=rep_universe, write_name=write_name) 


def main():
    logging.basicConfig(filename='csv_cleaner.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(message)s')
    rep_universe = create_rep_universe()

    for file in progress(glob.glob('./data/*.csv', recursive=True)):
        process_df(file, rep_universe)


if __name__ == "__main__":
    main()
