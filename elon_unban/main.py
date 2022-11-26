import json

DATA_PATH = "../../twitter_data/"
TEST_FILE = "./test.jsonl"


def main():
    with open("./twarc.log", "r") as fp:
        data = fp.read()

    print(data)


if __name__ == "__main__":
    main()
