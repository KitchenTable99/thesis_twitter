import json
import pickle
from typing import List, Set
import pandas as pd
import logging


HOUSE_115 = './data/representatives-accounts-1.csv'
HOUSE_116 = './data/congress116-house-accounts.csv'
SENATE_115 = './data/senators-accounts-1.csv'
SENATE_116 = './data/congress116-senate-accounts.csv'
HOUSE_117 = './data/flat_house_117.json'
SENATE_117 = './data/flat_senate_117.json'


def extract_ids(files: List[str]) -> Set[int]:
    to_return: Set[int] = set()
    for file in files:
        logging.info(f'Extracting from {file}')
        df: pd.DataFrame = pd.read_csv(file)

        ids: pd.Series = df['Uid']
        to_return.update(ids)
        logging.info(f'Added {ids.to_list()} to set of congressional ids')

    return to_return


def extract_json(file: str) -> Set[int]:
    to_return: Set[int] = set()
    logging.info('Extracting 117th')
    with open(file, 'r') as fp:
        data_117 = fp.read()

    json_117 = json.loads(data_117)
    for rep in json_117:
        try:
            twitter_id = rep.get('social').get('twitter_id')
            logging.info(f'Adding {twitter_id} to set of congressional_ids')
            to_return.add(twitter_id)
        except AttributeError:
            logging.fatal(f'{rep = } somehow fucked up')

    return to_return


def extract_twarc(files: List[str]) -> Set[int]:
    to_return: Set[int] = set()
    for file in files:
        with open(file, 'r') as fp:
            raw = fp.read()

        users = [json.loads(r) for r in raw.split('\n')[:-1]]
        for user in users:
            user_id = int(user.get('id'))
            logging.info(f'Adding {user_id}; {type(user_id)}')
            to_return.add(user_id)

    return to_return
            


def pickle_ids(congress_ids: Set[int], name: str) -> None:
    if not name.endswith('.pickle'):
        name += '.pickle'
    with open(name, 'wb') as fp:
        pickle.dump(congress_ids, fp)


def main():
    logging.basicConfig(filename='id_extractor.log', encoding='utf-8', level=logging.DEBUG)

    reps_115 = extract_ids([HOUSE_115, SENATE_115])
    reps_116 = extract_ids([HOUSE_116, SENATE_116])
    reps_117 = extract_twarc([HOUSE_117, SENATE_117])

    pickle_ids(reps_115, '115_congress')
    pickle_ids(reps_116, '116_congress')
    pickle_ids(reps_117, '117_congress')


if __name__ == "__main__":
    main()
