import pickle
from congressional_representatives import *
import pandas as pd
import glob
from clean_csv import create_rep_universe, process_df
from typing import Set, TextIO


def read_txt_source(fp: TextIO) -> Set[int]:
    raw_text = fp.read()
    lines = raw_text.split('\n')
    to_return = set()
    for line in lines:
        if line:
            to_return.add(int(line))

    return to_return


def get_difference():
    """
    Get the difference between the official stuff
    """
    with open('./source/downloaded_tweet_ids.pickle', 'rb') as fp:
        downloaded_set = pickle.load(fp)

    with open('./source/senators-1.txt', 'r') as fp:
        senators_115 = read_txt_source(fp)

    with open('./source/representatives-1.txt', 'r') as fp:
        reps_115 = read_txt_source(fp)

    with open('./source/house.txt', 'r') as fp:
        reps_116 = read_txt_source(fp)

    with open('./source/senate.txt', 'r') as fp:
        senators_116 = read_txt_source(fp)

    with open('./source/committees.txt', 'r') as fp:
        committees = read_txt_source(fp)

    requested_set = set()
    requested_set.update(senators_115)
    requested_set.update(senators_116)
    requested_set.update(committees)
    requested_set.update(reps_115)
    requested_set.update(reps_116)

    return requested_set - downloaded_set


def write_twarc_txt_file(attempt_download: Set[int]) -> None:
    to_write = ''
    for id in attempt_download:
        to_write += str(id)
        to_write += '\n'

    with open('download.txt', 'w') as fp:
        fp.write(to_write)


def get_successful_download_ids() -> Set[int]:
    to_return = set()
    for file in glob.glob('./out/*.gzip*'):
        df = pd.read_parquet(file)
        to_return.update(df['id'].to_list())

    return to_return


def main():
    rep_universe = create_rep_universe()
    num_existing_tries = len(glob.glob('./out/*.gzip*'))
    out_name = f'try_{num_existing_tries}'
    process_df('./temp/flat.csv', rep_universe, write_name=out_name)

    pre_unban_required = get_difference()
    victories = get_successful_download_ids()
    currently_required = pre_unban_required - victories
    print(victories in currently_required)
    print(len(currently_required), len(pre_unban_required), len(victories))
    write_twarc_txt_file(currently_required)


if __name__ == "__main__":
    main()
    print('done')
