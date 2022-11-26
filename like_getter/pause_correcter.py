import pickle
import os
from congressional_representatives import RepresentativeUniverse


def main():
    with open('pause.pickle', 'rb') as fp:
        paused: RepresentativeUniverse = pickle.load(fp)

    raw_users_done = os.listdir('./out/')
    users_done = [int(user.split('_')[0]) for user in raw_users_done]

    for user in users_done:
        paused.remove_rep(user)

    with open('pause.pickle', 'wb') as fp:
        pickle.dump(paused, fp)


if __name__ == "__main__":
    main()
