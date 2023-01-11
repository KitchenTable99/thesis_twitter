from dataclasses import dataclass, field
from datetime import datetime
import random
import pickle
from enum import Enum
from typing import Iterable, List, Optional, Tuple
from functools import total_ordering

@total_ordering
class Term(Enum):
    _115 = 115
    _116 = 116
    _117 = 117

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        else:
            return NotImplemented


@dataclass
class Representative:
    id: int
    terms: List[Term] = field(default_factory = list)

    def add_term(self, term: Term) -> None:
        self.terms.append(term)

    def get_total_term_boundaries(self) -> List[Tuple[datetime, datetime]]:
        """
        Returns a list of tuples with boundary conditions (inclusive, exclusive)
        """
        yes_115 = Term._115 in self.terms
        yes_116 = Term._116 in self.terms
        yes_117 = Term._117 in self.terms

        if yes_115 and not yes_116 and not yes_117:
            return [(datetime(2017, 1, 3), datetime(2019, 1, 3))]

        if not yes_115 and yes_116 and not yes_117:
            return [(datetime(2019, 1, 3), datetime(2021, 1, 3))]

        if not yes_115 and not yes_116 and yes_117:
            return [(datetime(2021, 1, 3), datetime(2022, 10, 28))]

        if yes_115 and yes_116 and not yes_117:
            return [(datetime(2017, 1, 3), datetime(2021, 1, 3))]

        if not yes_115 and yes_116 and yes_117:
            return [(datetime(2019, 1, 3), datetime(2022, 10, 28))]

        if yes_115 and not yes_116 and yes_117:
            return [(datetime(2017, 1, 3), datetime(2019, 1, 3)), ((datetime(2021, 1, 3), datetime(2022, 10, 28)))]

        if yes_115 and yes_116 and yes_117:
            return [(datetime(2017, 1, 3), datetime(2022, 10, 28))]

        return []


class RepresentativeUniverse:
    representatives: List[Representative]

    def __init__(self):
        self.representatives = []

    def contains(self, rep: Representative) -> bool:
        return rep in self.representatives

    def contains_id(self, id_: int) -> bool:
        for rep in self.representatives:
            if rep.id == id_:
                return True

        return False

    def get_rep(self, id_: int) -> Optional[Representative]:
        for rep in self.representatives:
            if rep.id == id_:
                return rep

        return None

    def add_rep(self, rep: Representative) -> None:
        self.representatives.append(rep)

    def remove_rep(self, id_: int) -> None:
        to_remove = self.get_rep(id_)
        if not to_remove:
            raise NotImplementedError
        self.representatives.remove(to_remove)
        
    def add_term_for_all(self, term: Term, rep_ids: Iterable[int]) -> None:
        for rep_id in rep_ids:
            target_rep = self.get_rep(rep_id)
            if not target_rep:
                target_rep = Representative(rep_id)
                self.add_rep(target_rep)

            target_rep.add_term(term)

    def pop_user_config(self) -> Tuple[int, Term]:
        rand_idx = random.randint(0, len(self.representatives) - 1)
        rand_rep = self.representatives[rand_idx]

        self.representatives.remove(rand_rep)

        return rand_rep.id, min(rand_rep.terms)

    def describe_representatives(self) -> None:
        n_115 = 0
        n_116 = 0
        n_117 = 0
        n_115_116 = 0
        n_116_117 = 0
        n_115_117 = 0
        n_all = 0

        for rep in self.representatives:
            yes_115 = Term._115 in rep.terms
            yes_116 = Term._116 in rep.terms
            yes_117 = Term._117 in rep.terms

            if yes_115 and not yes_116 and not yes_117:
                n_115 += 1

            if not yes_115 and yes_116 and not yes_117:
                n_116 += 1

            if not yes_115 and not yes_116 and yes_117:
                n_117 += 1

            if yes_115 and yes_116 and not yes_117:
                n_115_116 += 1

            if not yes_115 and yes_116 and yes_117:
                n_116_117 += 1

            if yes_115 and not yes_116 and yes_117:
                n_115_117 += 1

            if yes_115 and yes_116 and yes_117:
                n_all += 1


        print(f'{n_115 = }\n{n_116 = }\n{n_117 = }\n{n_115_116 = }\n{n_116_117 = }\n{n_115_117 = }\n{n_all = }')

    def __bool__(self) -> bool:
        return len(self.representatives) != 0

    def __repr__(self) -> str:
        to_return = str(self.representatives)
        return '\n'.join(to_return.split(', R'))


PICKLE_115 = './data/115_congress.pickle'
PICKLE_116 = './data/116_congress.pickle'
PICKLE_117 = './data/117_congress.pickle'


def main():
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

    # print(rep_universe)
    # rep_universe.describe_representatives()
    # for _ in range(10):
    #     print(rep_universe.pop_user_config())

    with open('rep_universe.pickle', 'wb') as fp:
        pickle.dump(rep_universe, fp)


if __name__ == "__main__":
    main()
