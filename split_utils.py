import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def train_val_test_split(df: pd.DataFrame, val_size=0.1, test_size=0.1, group_col="writer_id"):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(gss.split(df, groups=df[group_col]))

    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=43)
    train_idx, val_idx = next(gss2.split(train_val, groups=train_val[group_col]))

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    return train, val, test
