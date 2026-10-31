"""Can the UPI labels be learned at all? Engineered features + strong model + label-permutation test.

Usage: .venv/bin/python scripts/upi_rescue_test.py      (or import rescue_test from the notebook)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

AGE_MID = {'18-25': 21, '26-35': 30, '36-45': 40, '46-55': 50, '56+': 60}
CATS = ['transaction type', 'merchant_category', 'sender_age_group', 'receiver_age_group', 'sender_state',
        'sender_bank', 'receiver_bank', 'device_type', 'network_type', 'day_of_week']


def rescue_features(u):
    # Original columns (leakage excluded) plus 14 engineered ones that real or generated fraud often leaves behind
    ts = pd.to_datetime(u.timestamp)
    amt = u['amount (INR)']
    return pd.DataFrame({
        **{c: u[c].astype('category').cat.codes for c in CATS},
        'amount': amt, 'hour': u.hour_of_day, 'is_weekend': u.is_weekend,
        'log_amount': np.log1p(amt), 'round_100': (amt % 100 == 0).astype(int), 'round_500': (amt % 500 == 0).astype(int),
        'amount_last_digit': amt % 10, 'same_bank': (u.sender_bank == u.receiver_bank).astype(int),
        'age_gap': u.sender_age_group.map(AGE_MID) - u.receiver_age_group.map(AGE_MID),
        'minute': ts.dt.minute, 'second': ts.dt.second, 'day_of_year': ts.dt.dayofyear, 'month': ts.dt.month,
        'is_night': u.hour_of_day.between(0, 5).astype(int),
        'amount_vs_category_median': amt / u.groupby('merchant_category')['amount (INR)'].transform('median'),
        'amount_vs_type_median': amt / u.groupby('transaction type')['amount (INR)'].transform('median'),
        'txn_seq': u['transaction id'].str[3:].astype(int),   # row order: do labels follow the ID sequence?
    })


def rescue_test(u, n_shuffles=20, seed=42):
    # Score the real labels, then the same pipeline on randomly shuffled labels (the "no pattern" reference)
    X, y = rescue_features(u), u.fraud_flag.values
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)

    def score(labels):
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=seed)
        p = cross_val_predict(model, X, labels, cv=cv, method='predict_proba')[:, 1]
        return roc_auc_score(labels, p), average_precision_score(labels, p)

    real_auc, real_ap = score(y)
    rng = np.random.default_rng(seed)
    null = np.array([score(rng.permutation(y)) for _ in range(n_shuffles)])
    return {'n_features': X.shape[1], 'n_engineered': X.shape[1] - len(CATS) - 3,
            'real_roc_auc': real_auc, 'real_pr_auc': real_ap, 'random_pr_auc': y.mean(),
            'shuffled_roc_auc': null[:, 0], 'shuffled_pr_auc': null[:, 1],
            'p_value': (np.sum(null[:, 0] >= real_auc) + 1) / (n_shuffles + 1)}


if __name__ == '__main__':
    u = pd.read_csv(Path(__file__).resolve().parent.parent / 'upi_transactions 2024.csv')
    r = rescue_test(u)
    print(f"{r['n_features']} features ({r['n_engineered']} engineered)")
    print(f"REAL labels      ROC-AUC {r['real_roc_auc']:.4f}   PR-AUC {r['real_pr_auc']:.5f}   (random {r['random_pr_auc']:.5f})")
    print(f"SHUFFLED labels  ROC-AUC {r['shuffled_roc_auc'].mean():.4f} ± {r['shuffled_roc_auc'].std():.4f}")
    print(f"p-value {r['p_value']:.3f}")
