import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def cv_score(
    clf,
    X,
    y,
    sample_weight,
    scoring="neg_log_loss",
    t1: pd.Series = None,
    cv=None,
    cv_gen=None,
    pct_embargo: float = None,
):
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise ValueError("scoring must be neg_log_loss or accuracy")
    from sklearn.metrics import accuracy_score, log_loss

    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
    score = []
    for train, test in cv_gen.split(X=X):
        fit = clf.fit(
            X.iloc[train, :],
            y.iloc[train],
            sample_weight=sample_weight.iloc[train].values,
        )
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(
                y.iloc[test],
                prob,
                sample_weight=sample_weight.iloc[test].values,
                labels=clf.classes_,
            )
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(
                y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values
            )
        score.append(score_)
    return np.array(score)


class PurgedKFold(KFold):
    """
    기본 KFold를 상속받아서 purge를 포함한 KFold를 구현한 클래스
    """

    def __init__(self, n_splits=5, t1: pd.Series = None, pct_embargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series")
        super(PurgedKFold, self).__init__(
            n_splits=n_splits, shuffle=False, random_state=None
        )
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X.index must be same as t1.index")
        indices = np.arange(len(X))
        mbrg = int(len(X) * self.pct_embargo)
        test_starts = [
            (i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)
        ]
        for t_in, t_out in test_starts:
            test_start = self.t1.index[t_in]
            test_indices = indices[t_in:t_out]
            max_t1_idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= test_start].index
            )
            if max_t1_idx < len(X):  # right train with embargo
                train_indices = np.concatenate(
                    [train_indices, indices[max_t1_idx + mbrg :]]
                )
            yield train_indices, test_indices
