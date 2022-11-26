import json
import os
import signal
import logging
import pickle
from congressional_representatives import RepresentativeUniverse, Term
from datetime import datetime
from twarc.client2 import Twarc2
from twarc.expansions import ensure_flattened


PICKLE_115 = './data/115_congress.pickle'
PICKLE_116 = './data/116_congress.pickle'
PICKLE_117 = './data/117_congress.pickle'
STOP = False


def stop_handler(sig, frame):
    print('Stopping once finished with representative')
    global STOP
    STOP = True


def user_done(page, first_term: Term) -> bool:
    logging.debug(f'Checking tweet against firt term: {first_term}')
    match first_term:
        case Term._115:
            earliest_search = datetime(2017, 1, 3)
        case Term._116:
            earliest_search = datetime(2019, 1, 3)
        case Term._117:
            earliest_search = datetime(2021, 1, 3)
        case _:
            logging.warning('Some term was not specified')
            earliest_search = datetime(2022, 11, 2)
    logging.info(f'{earliest_search = }')

    earliest_tweet = datetime.fromisoformat(
            list(ensure_flattened(page))[-1]
            .get('created_at') 
            .split('T')[0]
        )

    logging.info(f'{earliest_tweet = }')

    return  earliest_search > earliest_tweet


def create_rep_universe() -> RepresentativeUniverse:
    if os.path.exists('pause.pickle'):
        with open('pause.pickle', 'rb') as fp:
            return pickle.load(fp)

    rep_universe = RepresentativeUniverse()

    with open(PICKLE_115, 'rb') as fp:
        congress_115 = pickle.load(fp)
    rep_universe.add_term_for_all(Term._115, congress_115)

    with open(PICKLE_116, 'rb') as fp:
        congress_116 = pickle.load(fp)
    rep_universe.add_term_for_all(Term._116, congress_116)

    with open(PICKLE_117, 'rb') as fp:
        congress_117 = pickle.load(fp)
    rep_universe.add_term_for_all(Term._117, congress_117)

    return rep_universe


def process_likes_page(user_id: int, page) -> None:
    with open(f'./out/{user_id}_likes.jsonl', 'a') as fp:
        fp.write(json.dumps(page) + '\n')


def main():
    logging.basicConfig(filename='likes.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(message)s')
    signal.signal(signal.SIGINT, stop_handler)
    client = Twarc2(bearer_token="AAAAAAAAAAAAAAAAAAAAAHNwiAEAAAAAF72%2Fj1mdKq00j%2FtZhNe6sL2yUng%3DdBSexd2M8w4mbWInSZm1EjEYt9FoES7FMyACFoBYlzXiiKNquZ")
    rep_universe = create_rep_universe()

    logging.info('Like gathering beginning now.')
    while rep_universe and not STOP:
        user_id, first_term = rep_universe.pop_user_config()
        logging.info(f'{len(rep_universe.representatives)} representatives remaining')
        logging.info(f'Gathering likes for user: {user_id}')

        for page_num, page in enumerate(client.liked_tweets(user_id)):
            if STOP:
                print('Stopping soon...')
            logging.info(f'Getting page {page_num}')
            process_likes_page(user_id, page)

            if user_done(page, first_term):
                logging.info(f'Reached terminal point for {user_id}')
                break

    if STOP:
        logging.info('Finishing due to keyboard interrupt')
        with open('pause.pickle', 'wb') as fp:
            pickle.dump(rep_universe, fp)

    logging.info('Done!')


if __name__ == "__main__":
    pass
