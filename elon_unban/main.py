import pickle
from typing import Set, TextIO


def read_txt_source(fp: TextIO) -> Set[int]:
    raw_text = fp.read()
    lines = raw_text.split('\n')
    to_return = set()
    for line in lines:
        if line:
            to_return.add(int(line))

    return to_return


def main():
    with open('./downloaded_tweet_ids.pickle', 'rb') as fp:
        downloaded_set = pickle.load(fp)

    with open('./senators-1.txt', 'r') as fp:
        senators_115 = read_txt_source(fp)

    with open('./representatives-1.txt', 'r') as fp:
        reps_115 = read_txt_source(fp)

    with open('./house.txt', 'r') as fp:
        reps_116 = read_txt_source(fp)

    with open('./senate.txt', 'r') as fp:
        senators_116 = read_txt_source(fp)

    with open('./committees.txt', 'r') as fp:
        committees = read_txt_source(fp)

    requested_set = set()
    requested_set.update(senators_115)
    requested_set.update(senators_116)
    requested_set.update(committees)
    requested_set.update(reps_115)
    requested_set.update(reps_116)

    prevented_set = requested_set - downloaded_set
    with open('difference.pickle', 'wb') as fp:
        pickle.dump(prevented_set, fp)

    print(len(prevented_set))




if __name__ == "__main__":
    main()
