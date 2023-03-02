import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def needs_normalization(col_name: str, col: pd.Series) -> bool:
    min_0 = col.min() == 0.
    max_1 = col.max() == 1.
    between_0_1 = min_0 and max_1

    is_percent = 'percent' in col_name

    return not between_0_1 and not is_percent


def main():
    df = pd.read_csv('trimmed.csv')

    knn = KNeighborsRegressor(n_neighbors=2)



    normalize_columns = [c for c in df.columns if needs_normalization(c, df[c])]
    df[normalize_columns] = (df[normalize_columns] - df[normalize_columns].mean()) / df[normalize_columns].std()

    train, test = train_test_split(df, test_size=.1)

    df.to_csv('normalized.csv', index=False)
    train.to_csv('train_normalized.csv', index=False)
    test.to_csv('test_normalized.csv', index=False)


if __name__ == "__main__":
    main()
