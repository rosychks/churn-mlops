"""CI tests: the data loads, the pipeline trains, and it predicts sanely.
These run in GitHub Actions on every push."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from preprocess import load_data, build_preprocessor

DATA = os.getenv("DATA_PATH", "data/telco_churn.csv")


def test_data_loads():
    X, y = load_data(DATA)
    assert len(X) == len(y) > 1000
    assert set(y.unique()) <= {0, 1}
    assert "customerID" not in X.columns


def test_pipeline_trains_and_beats_baseline():
    X, y = load_data(DATA)
    pre = build_preprocessor(X)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)
    pipe = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipe.fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    assert auc > 0.78, f"model regressed: ROC-AUC={auc:.3f}"