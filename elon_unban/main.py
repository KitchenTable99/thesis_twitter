import json

DATA_PATH = '../../twitter_data/'
TEST_FILE = './test.jsonl'


def main():
    # all_115 = DATA_PATH + '/115/reps_115.jsonl'
    with open(TEST_FILE, 'r') as fp:
        json_list = list(fp)

    username = {}
    reply = {}
    reference = {}
    id = {}
    for json_str in json_list:
        result = json.loads(json_str)
        errors = result.get('errors', None)
        if errors:
            for err in errors:
                print(err)


    print(counts)


if __name__ == "__main__":
    main()
