import pandas as pd
from one_hot import convert_int_code, make_one_hot_df
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from enum import Enum


class ColumnDesignation(Enum):
    NORMALIZED = 1,
    IGNORED = 2,
    ONE_HOT = 3


def get_ynh(col):
    user_input = input(f'Should this column be [n]ormalized, [i]gnored, or [o]ne-hot coded: {col}? ')
    if user_input.lower().startswith('n'):
        return ColumnDesignation.NORMALIZED
    elif user_input.lower().startswith('i'):
        return ColumnDesignation.IGNORED
    elif user_input.lower().startswith('o'):
        return ColumnDesignation.ONE_HOT
    else:
        print('You must answer with something beginning with [n/i/o]')
        return get_ynh(col)


def get_columns():
    df = pd.read_csv('testset_nn.csv')

    normalize = []
    ignore = []
    one_hot = []
    for column in sorted(df.columns.to_list()):
        designation = get_ynh(column)
        if designation == ColumnDesignation.NORMALIZED:
            normalize.append(column)
        if designation == ColumnDesignation.IGNORED:
            ignore.append(column)
        if designation == ColumnDesignation.ONE_HOT:
            one_hot.append(column)

    with open('normalize.pickle', 'wb') as fp:
        pickle.dump(normalize, fp)
    with open('ignore.pickle', 'wb') as fp:
        pickle.dump(ignore, fp)
    with open('one_hot.pickle', 'wb') as fp:
        pickle.dump(one_hot, fp)


def main():
    df = pd.read_csv('testset_nn.csv', sep=';')

    with open('normalize.pickle', 'rb') as fp:
        normalize_columns = pickle.load(fp)
    # df[normalize_columns] = (df[normalize_columns] - df[normalize_columns].mean()) / df[normalize_columns].std()

    with open('one_hot.pickle', 'rb') as fp:
        one_hot_columns = pickle.load(fp)
        one_hot_columns.remove('tweet__hour_of_day')
    oh_dfs = []
    for column in one_hot_columns:
        num_categories = len(df[column].unique()) + 1 if column != 'user__created_hour_of_day' else 24
        oh_matrix = convert_int_code(df[column].to_list(), num_categories)
        oh_dfs.append(make_one_hot_df(oh_matrix, column))

    oh_columns = pd.concat(oh_dfs, axis=1)
    df = pd.concat([df, oh_columns], axis=1)
    df = df.drop(columns=one_hot_columns)
    df = df.drop(columns=['tweet__is_withheld_copyright', 'tweet__url_only',
                          'user__has_url', 'user__is_english',
                          'user__more_than_50_tweets', 'tweet__possibly_sensitive_news',
                          'tweet__day_of_month_0', 'tweet__day_of_week_7',
                           'user__tweets_per_week', 'user__has_country',
                          'tweet__is_quote_status', 'tweet__user_id', 'tweet__hour_of_day'])
    df = df.rename(columns={'user__tweets_in_different_lang': 'user__nr_languages_tweeted'})

    needs_inference = df['tweet__possibly_sensitive'].isna()
    all_but_inference = df.loc[:, ~df.columns.isin(['tweet__possibly_sensitive'])]
    knn_train_x = all_but_inference[~needs_inference]
    knn_train_y = df[~needs_inference]['tweet__possibly_sensitive']
    knn_test_x = all_but_inference[needs_inference]

    knn = KNeighborsRegressor(n_neighbors=2)
    print(df.isna().any()[lambda x: x])
    knn.fit(knn_train_x, knn_train_y)
    knn_test_x['tweet__possibly_sensitive'] = knn.predict(knn_test_x)

    no_inference = df[~needs_inference]
    finished_inference = pd.concat([knn_test_x, no_inference])
    df = finished_inference
    df.to_csv('inferred.csv', index=False)

    # train, test = train_test_split(df, test_size=.1)
    #
    # train.to_csv('train_normalized.csv', index=False)
    # test.to_csv('test_normalized.csv', index=False)


if __name__ == "__main__":
    main()

