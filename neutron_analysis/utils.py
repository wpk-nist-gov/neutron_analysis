from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=10)
def read_table_once(path, **kws):
    return pd.read_csv(path, **kws)


def from_index_or_col(df, k):
    if k in df.index.names:
        v = df.index.get_level_values(k)
    else:
        v = df[k]
    return v


def ratio_table(df, index_cols=None, data_cols=None, std_val="std", filter_den=False):

    if data_cols is None:
        data_cols = list(df.columns)

    if index_cols is None:
        index_cols = list(df.index.names)

    if isinstance(data_cols, str):
        data_cols = [data_cols]

    if isinstance(index_cols, str):
        index_cols = [index_cols]

    df = df.reset_index().set_index(index_cols)[data_cols]

    num = df.query("{} == @std_val".format(index_cols[0]))

    if len(num) == 1:
        num = num.iloc[0]
    else:
        num = num.droplevel(index_cols[0])

    if filter_den:
        den = df.query("{} != @std_val".format(index_cols[0]))
    else:
        den = df

    out = num / den

    return out


def loc_levels(df, indexer, columns=None):
    """
    indexer = {level_name: vals, ....}
    """

    if columns is None:
        columns = slice(None)

    args = []
    for k in df.index.names:
        if k in indexer:
            v = indexer[k]
        else:
            v = slice(None)

        args.append(v)

    return df.loc[pd.IndexSlice[tuple(args)], columns]


def keys_in_names(keys, *names):

    s = set()
    for name in names:
        s = s.union(set(name))

    if isinstance(keys, str):
        keys = [keys]

    for k in keys:
        if k not in s:
            raise ValueError(f"{k} not in {s}")


def reorder_names(names_supplied, names_all):
    """
    Resolves a supplied list containing an ellispsis representing other items, to
    a generator with the 'realized' list of all items
    """
    if ... in names_supplied:
        if len(set(names_all)) != len(names_all):
            raise ValueError("Cannot use ellipsis with repeated names")
        if len([d for d in names_supplied if d == ...]) > 1:
            raise ValueError("More than one ellipsis supplied")
        other_names = [d for d in names_all if d not in names_supplied]
        for d in names_supplied:
            if d == ...:
                yield from other_names
            else:
                yield d
    else:
        if set(names_supplied) ^ set(names_all):
            raise ValueError(
                f"{names_supplied} must be a permuted list of {names_all}, unless `...` is included"
            )
        yield from names_supplied


def reorder_index_levels(df, order):

    new_order = list(reorder_names(order, df.index.names))

    out = df.copy()
    out.index = out.index.reorder_levels(new_order)

    return out
