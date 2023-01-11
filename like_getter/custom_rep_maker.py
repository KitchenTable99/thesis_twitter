from congressional_representatives import RepresentativeUniverse, Term
from typing import List


def get_int_list(path: str) -> List[int]:
    with open(path, 'r') as fp:
        data = fp.read()
    split_list = data.split('\n')
    return [int(item) for item in split_list if item]


def main():
    dummy_universe = RepresentativeUniverse()

    target_ids = get_int_list(PATH)
    dummy_universe.add_term_for_all(Term._115, target_ids)


if __name__ == "__main__":
    main()
