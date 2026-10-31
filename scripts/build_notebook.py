"""Builds notebooks/fraud_analysis.ipynb from the cells below."""
import sys
import nbformat as nbf

OUT = sys.argv[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))


def step(title, what, why, how, impact):
    # A "What / Why / How / Impact" summary box with the measured evidence for one step
    md(f"""
### 📌 {title}

| | |
|---|---|
| **What** we did | {what} |
| **Why** | {why} |
| **How** | {how} |
| **Impact (evidence)** | {impact} |

<sub>Numbers come from the seed-42 run with the placeholder costs (₹X = 10,000, ₹Y = 500, ₹Z = 10). Re-run after the official costs are known.</sub>
""")

md("""
# 🏦 Real-Time UPI / Digital Payment Fraud Detection

**KLU+Path-3 · ML-02 Hackathon**

One fraud decision system, tested on **two payment channels**:

| Channel | Dataset | What it is |
|---|---|---|
| **UPI** | `upi_transactions 2024.csv` | 250,000 UPI payments, with readable fields such as type, merchant, bank, device and amount in ₹. The brief describes it as *simulated* |
| **Card / digital** | `creditcard.csv` | 284,807 real card payments (ULB). The features are anonymised by PCA |

| Section | Problem statement |
|---|---|
| 1 | PS1 - Data quality and reliability (both channels) |
| 2 | PS2 - The fraud lens and the accuracy trap |
| 3 | PS3 - Fraud patterns (EDA) for both channels, plus a UPI signal test |
| 4 | PS4 - Supervised models (LR, RF, XGBoost) × (no handling, class weights, SMOTE) against a rule baseline: card channel |
| 5 | PS5 - Anomaly detection without labels (Isolation Forest): card channel |
| 6 | PS6 - Choosing a cut-off by cost, plus a sensitivity check: card channel |
| 7 | PS7 - The Approve / Step-up OTP / Block rule: card channel |
| 8 | PS8 - Final model and explanation (SHAP): card channel |
| 9 | **PS4-PS8 repeated on the UPI channel** with the identical pipeline |
| 10 | **The two channels side by side** |

**Rules followed:** a fixed seed (42), a time-aware split *before* any resampling, SMOTE only inside an `imblearn` pipeline (so it only ever sees training rows), all thresholds tuned on the **validation** slice and reported once on the **test** slice, and no leakage columns.
""")

code(r"""
import warnings, time, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score, precision_recall_curve,
                             confusion_matrix)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap, joblib

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA_CC, DATA_UPI = ROOT / 'creditcard.csv', ROOT / 'upi_transactions 2024.csv'
FIG, TAB, MOD = ROOT / 'outputs' / 'figures', ROOT / 'outputs' / 'tables', ROOT / 'models'
for d in (FIG, TAB, MOD):
    d.mkdir(parents=True, exist_ok=True)

# ⚠️ PLACEHOLDER COSTS: replace with the organiser-published values, then re-run everything
COST_MISSED_FRAUD = 10_000   # ₹X: one fraud that gets approved
COST_FALSE_BLOCK  = 500      # ₹Y: one genuine payment that gets blocked
COST_OTP          = 10       # ₹Z: one step-up OTP challenge (SMS + friction)
OTP_STOP_RATE     = 0.80     # ASSUMPTION: share of fraud stopped by the OTP step
MAX_OTP_SHARE     = 0.05     # FRICTION CAP: never send more than 5% of payments to OTP

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110
GOOD, BAD = '#2b7bba', '#d1495b'

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f'{name}.png', bbox_inches='tight')
    plt.show()

def wilson(k, n, z=1.96):
    # 95% confidence interval for a rate; shows how much a fraud rate can wobble by chance
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

print('costs: missed fraud ₹%s | false block ₹%s | OTP ₹%s | OTP stop rate %.0f%%'
      % (COST_MISSED_FRAUD, COST_FALSE_BLOCK, COST_OTP, OTP_STOP_RATE * 100))
""")

# ---------------------------------------------------------------- PS1
md("""
## 1 · Data quality and reliability (PS1)

We check both files for missing values, duplicates, impossible amounts, inconsistent categories, broken timestamps and **leakage columns**. A leakage column is information that is only known *after* the fraud outcome.
""")

code(r"""
cc_raw = pd.read_csv(DATA_CC)
upi_raw = pd.read_csv(DATA_UPI)
print('creditcard:', cc_raw.shape, '| upi:', upi_raw.shape)

ts = pd.to_datetime(upi_raw['timestamp'], errors='coerce')
obj_cols = upi_raw.select_dtypes('object').columns.drop(['transaction id', 'timestamp'])
messy_cats = sum(upi_raw[c].nunique() != upi_raw[c].str.strip().str.lower().nunique() for c in obj_cols)
dup_cc = cc_raw.duplicated()

dq = pd.DataFrame([
    ['creditcard', 'Missing values', cc_raw.isna().sum().sum(), 'Nothing found', 'No action'],
    ['creditcard', 'Exact duplicate rows', dup_cc.sum(), f'{cc_raw[dup_cc].Class.sum()} of them are fraud',
     'DROPPED (keep first): copies inflate scores and can sit on both sides of the split'],
    ['creditcard', 'Amount = 0', (cc_raw.Amount == 0).sum(), 'Plausible card-verification pings',
     'KEPT; noted in limitations'],
    ['creditcard', 'Amount < 0', (cc_raw.Amount < 0).sum(), 'Nothing found', 'No action'],
    ['creditcard', 'Time out of order / negative', (cc_raw.Time.diff() < 0).sum() + (cc_raw.Time < 0).sum(),
     f'Time runs 0 to {cc_raw.Time.max():,.0f}s (about 2 days)', 'Used to make the time-aware split and the hour of day'],
    ['creditcard', 'V1 to V28 (anonymised PCA)', 28, 'Real meaning is hidden', 'Used as features; limits how far we can explain them'],
    ['upi', 'Missing values', upi_raw.isna().sum().sum(), 'Nothing found', 'No action'],
    ['upi', 'Duplicate rows / duplicate IDs', upi_raw.duplicated().sum() + upi_raw['transaction id'].duplicated().sum(), 'Nothing found', 'No action'],
    ['upi', 'Unparseable timestamps', ts.isna().sum(), f'Range {ts.min():%Y-%m-%d} to {ts.max():%Y-%m-%d}', 'No action'],
    ['upi', 'hour / day / weekend that disagree with the timestamp',
     (ts.dt.hour != upi_raw.hour_of_day).sum() + (ts.dt.day_name() != upi_raw.day_of_week).sum()
     + ((ts.dt.dayofweek >= 5).astype(int) != upi_raw.is_weekend).sum(),
     'These are derived copies of the timestamp', 'Kept (consistent)'],
    ['upi', 'Amount <= 0', (upi_raw['amount (INR)'] <= 0).sum(),
     f"₹{upi_raw['amount (INR)'].min()} to ₹{upi_raw['amount (INR)'].max():,}", 'No action'],
    ['upi', 'Categories with case or space variants', messy_cats, 'Nothing found', 'No action'],
    ['upi', 'transaction id', upi_raw['transaction id'].nunique(), 'Pure identifier', 'EXCLUDED (no signal, memorisable)'],
    ['upi', 'transaction_status', (upi_raw.transaction_status == 'FAILED').sum(),
     'SUCCESS/FAILED is only known AFTER processing', 'EXCLUDED as LEAKAGE'],
], columns=['dataset', 'check', 'count', 'finding', 'treatment'])
dq.to_csv(TAB / 'data_quality.csv', index=False)
dq
""")

code(r"""
# Clean creditcard: drop exact duplicates, then sort by time for the time-aware split
cc = cc_raw.drop_duplicates().sort_values('Time', kind='stable').reset_index(drop=True)
cc['hour'] = (cc.Time // 3600 % 24).astype(int)
print(f'creditcard after cleaning: {len(cc):,} rows, {cc.Class.sum()} frauds')

upi = upi_raw.assign(timestamp=ts).sort_values('timestamp', kind='stable').reset_index(drop=True)
UPI_LEAKAGE = ['transaction id', 'transaction_status']
""")

step('Step 1 summary · Data quality',
     'Checked both files for 13 kinds of problems, removed duplicate rows and set aside the "cheat" (leakage) columns.',
     'If the data is wrong, every result after it is wrong. A duplicate row can land in both train and test, so the model gets "tested" on a row it has already seen and looks better than it really is.',
     'pandas checks: `isna`, `duplicated`, value ranges, parsing every timestamp, re-deriving hour/day/weekend from the timestamp, and a case/space check on categories. Duplicates are dropped; `transaction id` and `transaction_status` are excluded.',
     'creditcard: 284,807 → **283,726 rows** (1,081 duplicates dropped, 19 of them fraud), so **473 frauds** remain. '
     'UPI: 0 missing, 0 duplicates, 0 bad timestamps. That is *perfectly* clean, which is unusual for real payment data and a first hint that it is synthetic. '
     '`transaction_status` (12,376 FAILED) is excluded because it is only known after processing. Full table: `outputs/tables/data_quality.csv`.')

# ---------------------------------------------------------------- PS2
md("""
## 2 · The fraud lens and the accuracy trap (PS2)

- **Target:** `Class` (credit card) / `fraud_flag` (UPI), where 1 = fraud.
- **Split:** time-aware. The oldest 60% of transactions are for **train**, the next 20% for **validation** (used to pick cut-offs and the final model) and the newest 20% for **test** (touched once, for reporting). Real systems predict the future from the past, so the split follows time.
""")

code(r"""
counts = pd.DataFrame({
    'dataset': ['creditcard', 'upi'],
    'transactions': [len(cc), len(upi)],
    'frauds': [cc.Class.sum(), upi.fraud_flag.sum()],
})
counts['genuine'] = counts.transactions - counts.frauds
counts['fraud_rate_%'] = (counts.frauds / counts.transactions * 100).round(3)
counts['genuine_per_fraud'] = (counts.genuine / counts.frauds).round(0)
display(counts)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
for a, (name, y) in zip(ax, [('creditcard', cc.Class), ('UPI 2024', upi.fraud_flag)]):
    vc = y.value_counts().sort_index()
    a.bar(['Genuine', 'Fraud'], vc.values, color=[GOOD, BAD])
    a.set_yscale('log')
    a.set_title(f'{name}: {vc[1]:,} fraud vs {vc[0]:,} genuine (log scale)')
    for i, v in enumerate(vc.values):
        a.text(i, v, f'{v:,}', ha='center', va='bottom')
savefig('01_class_imbalance')
""")

code(r"""
FEATURES = [f'V{i}' for i in range(1, 29)] + ['log_amount', 'hour_sin', 'hour_cos']

def cc_features(df):
    X = df[[f'V{i}' for i in range(1, 29)]].copy()
    X['log_amount'] = np.log1p(df.Amount)
    X['hour_sin'] = np.sin(2 * np.pi * df.hour / 24)
    X['hour_cos'] = np.cos(2 * np.pi * df.hour / 24)
    return X

n = len(cc)
i1, i2 = int(n * 0.6), int(n * 0.8)
train, val, test = cc.iloc[:i1], cc.iloc[i1:i2], cc.iloc[i2:]
X_tr, X_va, X_te = cc_features(train), cc_features(val), cc_features(test)
y_tr, y_va, y_te = train.Class.values, val.Class.values, test.Class.values

split = pd.DataFrame({
    'slice': ['train', 'validation', 'test'],
    'rows': [len(train), len(val), len(test)],
    'frauds': [y_tr.sum(), y_va.sum(), y_te.sum()],
    'time_from_h': [train.Time.min() / 3600, val.Time.min() / 3600, test.Time.min() / 3600],
    'time_to_h': [train.Time.max() / 3600, val.Time.max() / 3600, test.Time.max() / 3600],
}).round(1)
split['fraud_rate_%'] = (split.frauds / split.rows * 100).round(3)
split
""")

code(r"""
# The do-nothing model: say "genuine" for every transaction
dummy = DummyClassifier(strategy='most_frequent').fit(X_tr, y_tr)
p = dummy.predict(X_te)
trap = pd.DataFrame([{
    'model': 'Do-nothing (always genuine)',
    'accuracy_%': accuracy_score(y_te, p) * 100,
    'recall_%': recall_score(y_te, p) * 100,
    'precision_%': precision_score(y_te, p, zero_division=0) * 100,
    'frauds_caught': int(((p == 1) & (y_te == 1)).sum()),
    'frauds_missed': int(((p == 0) & (y_te == 1)).sum()),
    'expected_cost_₹': int(((p == 0) & (y_te == 1)).sum() * COST_MISSED_FRAUD),
}]).round(3)
trap
""")

md("""
**The accuracy trap:** the do-nothing model looks more than 99.8% "accurate" and catches **zero** frauds. Accuracy rewards the model for the 99.8% of easy "genuine" answers and hides exactly the thing we care about.

**Metrics we use instead:**
- **Recall:** the share of fraud we stop.
- **Precision:** the share of our alarms that are real fraud.
- **F1:** combines precision and recall into one number.
- **PR-AUC:** a ranking quality score across all cut-offs. This is our main model-selection metric.
- **ROC-AUC:** reported, but with a warning that it looks optimistic when fraud is rare.
- **Expected ₹ cost:** the final business number.
""")

step('Step 2 summary · Fraud lens and the accuracy trap',
     'Measured how rare fraud is, split the data by time into train/validation/test, and built a "do-nothing" model.',
     'To prove that accuracy is useless here, and to fix our honest metrics *before* modelling so we cannot cherry-pick later. The time split copies real life, where we predict tomorrow from yesterday.',
     'Oldest 60% for train, next 20% for validation, newest 20% for test. `DummyClassifier(strategy="most_frequent")` always answers "genuine".',
     'Fraud is **0.167%** of transactions (about 1 in 600). The do-nothing model scores **99.87% accuracy** and catches **0 of 74** test frauds, which means a ₹7.4 lakh loss on the test slice. '
     'Our final model (Section 8) *also* shows 99.9% accuracy but catches 78% of the fraud, so accuracy cannot tell the useless model from the useful one. Charts: `01_class_imbalance.png`, `10_accuracy_trap_and_confusion.png`.')

# ---------------------------------------------------------------- PS3
md("""
## 3 · Fraud patterns (PS3)

The patterns below are **associations, not causes**. Each fraud rate has a 95% confidence interval. If two intervals overlap a lot, the difference may just be chance.
""")

code(r"""
def rate_table(df, col, target):
    g = df.groupby(col, observed=True)[target].agg(frauds='sum', n='count')
    g['rate_%'] = g.frauds / g.n * 100
    ci = np.array([wilson(k, m) for k, m in zip(g.frauds, g.n)]) * 100
    g['ci_low_%'], g['ci_high_%'] = ci[:, 0], ci[:, 1]
    return g

def plot_rates(ax, g, title, overall):
    x = np.arange(len(g))
    ax.bar(x, g['rate_%'], color=GOOD)
    ax.errorbar(x, g['rate_%'], yerr=[g['rate_%'] - g['ci_low_%'], g['ci_high_%'] - g['rate_%']],
                fmt='none', ecolor='black', capsize=3, lw=1)
    ax.axhline(overall, color=BAD, ls='--', lw=1, label=f'overall {overall:.3f}%')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in g.index], rotation=45, ha='right')
    ax.set_ylabel('fraud rate %')
    ax.set_title(title)
    ax.legend()

cc['amount_band'] = pd.cut(cc.Amount, [-0.01, 0, 1, 10, 50, 200, 1000, 1e9],
                           labels=['0', '0-1', '1-10', '10-50', '50-200', '200-1k', '>1k'])
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
cc_hour = rate_table(cc, 'hour', 'Class')
cc_amt = rate_table(cc, 'amount_band', 'Class')
plot_rates(ax[0], cc_hour, 'creditcard: fraud rate by hour of day', cc.Class.mean() * 100)
plot_rates(ax[1], cc_amt, 'creditcard: fraud rate by amount band', cc.Class.mean() * 100)
savefig('02_cc_fraud_by_hour_amount')
display(cc_amt.round(3))
print(cc.groupby('Class').Amount.describe().round(2))
""")

code(r"""
upi['amount_band'] = pd.cut(upi['amount (INR)'], [0, 500, 1000, 2000, 5000, 10000, 1e9],
                            labels=['≤500', '500-1k', '1k-2k', '2k-5k', '5k-10k', '>10k'])
overall = upi.fraud_flag.mean() * 100
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
plot_rates(ax[0], rate_table(upi, 'transaction type', 'fraud_flag'), 'UPI: fraud rate by transaction type', overall)
plot_rates(ax[1], rate_table(upi, 'amount_band', 'fraud_flag'), 'UPI: fraud rate by amount band (₹)', overall)
plot_rates(ax[2], rate_table(upi, 'hour_of_day', 'fraud_flag'), 'UPI: fraud rate by hour of day', overall)
savefig('03_upi_fraud_by_type_amount_hour')

# Chi-square independence test: a low p-value means the fraud rate really differs across the groups
UPI_CATS = ['transaction type', 'merchant_category', 'amount_band', 'sender_age_group', 'receiver_age_group',
            'sender_state', 'sender_bank', 'receiver_bank', 'device_type', 'network_type',
            'hour_of_day', 'day_of_week', 'is_weekend']
chi = pd.DataFrame([(c, upi[c].nunique(), chi2_contingency(pd.crosstab(upi[c], upi.fraud_flag))[1])
                    for c in UPI_CATS], columns=['column', 'groups', 'p_value']).sort_values('p_value')
chi['differs_at_5%'] = chi.p_value < 0.05
# 13 tests at once: by luck alone about 1 in 20 will look "significant", so we use the Bonferroni limit 0.05 / 13
chi['significant_after_bonferroni'] = chi.p_value < 0.05 / len(chi)
chi.to_csv(TAB / 'upi_chi_square.csv', index=False)
chi.round(4)
""")

step('Step 3 summary · Where fraud lives (EDA)',
     'Fraud rate by hour and by amount band (credit card), and by transaction type, amount and hour (UPI), each with 95% confidence bars. Chi-square tests on 13 UPI columns.',
     'To find the risky periods and amounts that guide the rule baseline, the features and the recommendations, and to check whether the UPI patterns are real or just noise.',
     'A group-by of the fraud rate, Wilson 95% intervals, and `chi2_contingency` with a Bonferroni correction because we run 13 tests at once.',
     '**Night (1-5 AM): 0.67% fraud rate vs 0.14% at other hours (about 5×)**, peaking at 1.45% at 2 AM. Night holds 23% of all fraud. '
     '**Amount is U-shaped:** ₹0 amounts 1.38%, 0-1 amounts 0.51%, 10-50 only 0.06%, then up again to about 0.3% above 200. Small "card-testing" payments and big hits are both risky. Median fraud amount is 9.82 vs 22 for genuine. '
     '**UPI:** every bar overlaps the overall line. Only `sender_age_group` reaches p = 0.026, and that disappears after the Bonferroni correction (the limit is 0.0038). These are *associations*, not causes.')

md("""
### 3b · Can any model learn from the UPI labels?

We give the UPI data a fair chance. It gets the same time-aware split and leakage exclusions, one-hot encoded features, and a strong model plus a simple one. If the labels carry signal, PR-AUC should be clearly **above the fraud rate**, which is what random guessing scores.
""")

code(r"""
upi_X = pd.get_dummies(upi.drop(columns=UPI_LEAKAGE + ['timestamp', 'fraud_flag', 'amount_band']),
                       columns=[c for c in UPI_CATS if c not in ('amount_band', 'hour_of_day', 'is_weekend')],
                       dtype=float)
upi_X['log_amount'] = np.log1p(upi_X.pop('amount (INR)'))
upi_y = upi.fraud_flag.values
u1, u2 = int(len(upi) * 0.6), int(len(upi) * 0.8)
upi_spw = (upi_y[:u1] == 0).sum() / upi_y[:u1].sum()

upi_models = {
    'Logistic Regression (class weights)': Pipeline([('sc', StandardScaler()),
        ('m', LogisticRegression(max_iter=2000, class_weight='balanced'))]),
    'XGBoost (scale_pos_weight)': XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=upi_spw, tree_method='hist',
        random_state=SEED, n_jobs=-1, eval_metric='aucpr'),
}
rows = []
for name, m in upi_models.items():
    m.fit(upi_X.iloc[:u1], upi_y[:u1])
    s = m.predict_proba(upi_X.iloc[u2:])[:, 1]
    yt = upi_y[u2:]
    top = s >= np.quantile(s, 0.99)
    rows.append({'model': name, 'test_ROC_AUC': roc_auc_score(yt, s), 'test_PR_AUC': average_precision_score(yt, s),
                 'random_PR_AUC (= fraud rate)': yt.mean(), 'fraud_in_top_1%_%': yt[top].sum() / yt.sum() * 100})
upi_signal = pd.DataFrame(rows).round(4)
upi_signal.to_csv(TAB / 'upi_signal_test.csv', index=False)
upi_signal
""")

md("""
**UPI finding:** both models score about a coin flip (ROC-AUC ≈ 0.5), and PR-AUC sits at the random baseline. Together with the chi-square tests, this says the UPI `fraud_flag` is **not linked to any recorded column**: it looks randomly assigned. That makes it a **data-quality finding**. The rest of the modelling uses `creditcard.csv`, where a real signal exists. The UPI data is still used for business context.
""")

step('Step 3b summary · UPI signal test',
     'Gave the UPI data a fair modelling test before deciding not to use it for the main model.',
     'You should *prove* a dataset is unusable, not assume it. If we had trained on it blindly, every later chart would have been meaningless.',
     'The same time-aware 60/20/20 split, leakage columns excluded, one-hot encoding, then Logistic Regression and XGBoost.',
     'Test **ROC-AUC 0.54 / 0.51** (0.5 is a coin flip). PR-AUC is **0.0024 / 0.0022 against a random baseline of 0.0019**. The top 1% of scores holds 1-2% of the fraud, the same as random. '
     '**Decision:** the main model is built on `creditcard.csv`, and the UPI label problem becomes management recommendation #3.')

# ---------------------------------------------------------------- PS4
md("""
## 4 · Supervised models (PS4)

There are 3 algorithms × 3 imbalance settings, plus a hand-written rule baseline:

- **None:** the plain model.
- **Class weights:** each fraud counts about 600 times more in the loss (`class_weight='balanced'` / `scale_pos_weight`).
- **SMOTE:** synthetic fraud rows are created up to 10% of the genuine count. This happens inside an `imblearn` Pipeline, so it only runs during `fit` on **train** rows. Validation and test rows are never resampled.

Every model is fitted on train only. Settings are deliberately light: no tuning on test, and only modest tuning overall, because the time budget matters more than small gains.
""")

code(r"""
neg, pos = (y_tr == 0).sum(), y_tr.sum()
SPW = neg / pos
lr = lambda **kw: LogisticRegression(max_iter=3000, **kw)
rf = lambda **kw: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED, **kw)
xgb = lambda **kw: XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, tree_method='hist', random_state=SEED, n_jobs=-1,
                                 eval_metric='aucpr', **kw)
smote = lambda: SMOTE(sampling_strategy=0.1, random_state=SEED)

def make_models(spw):
    # The same 9 models for every channel; spw = genuine ÷ fraud in that channel's training slice
    return {
        ('Logistic Regression', 'No handling'):   Pipeline([('sc', StandardScaler()), ('m', lr())]),
        ('Logistic Regression', 'Class weights'): Pipeline([('sc', StandardScaler()), ('m', lr(class_weight='balanced'))]),
        ('Logistic Regression', 'SMOTE'):         Pipeline([('sc', StandardScaler()), ('smote', smote()), ('m', lr())]),
        ('Random Forest', 'No handling'):         rf(),
        ('Random Forest', 'Class weights'):       rf(class_weight='balanced_subsample'),
        ('Random Forest', 'SMOTE'):               Pipeline([('smote', smote()), ('m', rf())]),
        ('XGBoost', 'No handling'):               xgb(),
        ('XGBoost', 'Class weights'):             xgb(scale_pos_weight=spw),
        ('XGBoost', 'SMOTE'):                     Pipeline([('smote', smote()), ('m', xgb())]),
    }

MODELS = make_models(SPW)

scores_va, scores_te, fit_times = {}, {}, {}
for key, model in MODELS.items():
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_times[key] = time.time() - t0
    scores_va[key] = model.predict_proba(X_va)[:, 1]
    scores_te[key] = model.predict_proba(X_te)[:, 1]
    print(f'{key[0]:<20} {key[1]:<14} fit {fit_times[key]:6.1f}s  val PR-AUC {average_precision_score(y_va, scores_va[key]):.3f}')
""")

md("""
**Rule baseline (no ML):** this uses only readable fields, the way a fraud analyst might write it with no model. The EDA showed fraud is over-represented at night (about 1-5 AM) and among tiny "card-testing" amounts. The rule gives 1 point for each of those signs (0-2 points) and **flags a payment when it has 1 or more points**.
""")

code(r"""
def rule_score(df):
    night = df.hour.between(1, 5)
    tiny = df.Amount <= 1
    return (night.astype(int) + tiny.astype(int)) / 2   # 0, 0.5 or 1

RULE_KEY = ('Rule baseline', 'No ML')
scores_va[RULE_KEY], scores_te[RULE_KEY] = rule_score(val).values, rule_score(test).values
fit_times[RULE_KEY] = 0.0
""")

code(r"""
def cost_of(y, pred):
    fn = ((pred == 0) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    return fn * COST_MISSED_FRAUD + fp * COST_FALSE_BLOCK

def best_threshold(y, s, grid=None):
    # Cut-off with the lowest expected cost (block if score >= threshold)
    grid = np.unique(np.r_[np.linspace(0, 1, 1001), np.quantile(s, np.linspace(0.95, 1, 501))]) if grid is None else grid
    costs = np.array([cost_of(y, (s >= t).astype(int)) for t in grid])
    return grid[costs.argmin()], grid, costs

def evaluate(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'PR_AUC': average_precision_score(y, s), 'ROC_AUC': roc_auc_score(y, s),
            'precision': precision_score(y, pred, zero_division=0), 'recall': recall_score(y, pred),
            'F1': f1_score(y, pred, zero_division=0), 'accuracy': accuracy_score(y, pred),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'expected_cost_₹': fn * COST_MISSED_FRAUD + fp * COST_FALSE_BLOCK}

rows, best_thr = [], {}
for key in scores_te:
    default = 0.5 if key != RULE_KEY else 0.5          # rule: flag with 1 or more points
    t_cost, _, _ = best_threshold(y_va, scores_va[key])  # chosen on VALIDATION
    best_thr[key] = t_cost
    va_cost = cost_of(y_va, (scores_va[key] >= t_cost).astype(int))
    for label, t in [('default 0.5', default), ('cost-optimal', t_cost)]:
        rows.append({'model': key[0], 'imbalance': key[1], 'threshold_rule': label, 'threshold': t,
                     'val_PR_AUC': average_precision_score(y_va, scores_va[key]), 'val_cost_at_opt_₹': va_cost,
                     'fit_s': fit_times[key], **evaluate(y_te, scores_te[key], t)})
comparison = pd.DataFrame(rows)
comparison.to_csv(TAB / 'model_comparison.csv', index=False)

show = ['model', 'imbalance', 'threshold_rule', 'threshold', 'val_PR_AUC', 'PR_AUC', 'ROC_AUC',
        'precision', 'recall', 'F1', 'TP', 'FP', 'FN', 'expected_cost_₹', 'fit_s']
(comparison[show].round(3).style
    .background_gradient(subset=['PR_AUC'], cmap='Blues')
    .background_gradient(subset=['expected_cost_₹'], cmap='Reds_r')
    .format({'threshold': '{:.3f}', 'expected_cost_₹': '₹{:,.0f}', 'fit_s': '{:.1f}'}))
""")

code(r"""
pivot = comparison[comparison.threshold_rule == 'cost-optimal'].pivot(index='model', columns='imbalance', values='PR_AUC')
pivot_cost = comparison[comparison.threshold_rule == 'cost-optimal'].pivot(index='model', columns='imbalance', values='expected_cost_₹')
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
sns.heatmap(pivot.drop(index='Rule baseline').drop(columns='No ML'), annot=True, fmt='.3f', cmap='Blues', ax=ax[0], cbar=False)
ax[0].set_title('Test PR-AUC: algorithm × imbalance method')
sns.heatmap(pivot_cost.drop(index='Rule baseline').drop(columns='No ML') / 1000, annot=True, fmt='.0f', cmap='Reds', ax=ax[1], cbar=False)
ax[1].set_title('Test expected cost (₹ thousands) at the cost-optimal cut-off')
savefig('04_imbalance_method_comparison')
""")

step('Step 4 summary · Supervised models',
     'Trained 9 models (Logistic Regression, Random Forest and XGBoost × no handling, class weights and SMOTE) and compared them with a hand-written night/tiny-amount rule.',
     'The brief requires it, and we wanted to learn *whether imbalance handling really helps* instead of assuming it does.',
     'Every model is fitted on train only. SMOTE (fraud up to 10% of genuine) runs inside an `imblearn` Pipeline, so it only touches training rows. Class weights use `balanced` / `scale_pos_weight` ≈ 600. Each model\'s cut-off is chosen on validation and reported on test.',
     '**Test PR-AUC:** Random Forest 0.79-0.82, XGBoost 0.78-0.80, Logistic Regression 0.70-0.75. '
     '**The rule baseline fails:** it catches 38% of fraud but raises 6,343 false alarms (precision 0.4%), which costs ₹36 lakh, *5× worse than doing nothing*. '
     '**Imbalance handling** lifts Logistic Regression a lot (validation PR-AUC 0.61 → 0.75 with class weights) but barely matters for tree models once the cut-off is tuned by cost. Class weights beat SMOTE in 2 of 3 algorithms. '
     '**The default 0.5 cut-off is dangerous:** Logistic Regression with class weights at 0.5 blocks 2,921 genuine payments (₹15.3 lakh), while the same model at its cost cut-off costs ₹2.0 lakh. '
     '**Training time:** XGBoost about 3 s, Random Forest 30-50 s, Logistic Regression under 1 s.')

# ---------------------------------------------------------------- PS5
md("""
## 5 · Anomaly detection without labels (PS5)

New fraud tricks have no labels when they first appear. An **Isolation Forest** is trained on the training features with **no `Class` column at all**. It isolates unusual points with random cuts: the fewer cuts a point needs, the more anomalous it is. Labels are used **only afterwards**, to check how much fraud ends up in the most anomalous transactions.
""")

code(r"""
t0 = time.time()
iso = IsolationForest(n_estimators=300, max_samples=256, random_state=SEED, n_jobs=-1).fit(X_tr)  # no y!
iso_time = time.time() - t0
iso_va, iso_te = -iso.score_samples(X_va), -iso.score_samples(X_te)   # higher = more anomalous
ISO_KEY = ('Isolation Forest', 'No labels')
scores_va[ISO_KEY], scores_te[ISO_KEY], fit_times[ISO_KEY] = iso_va, iso_te, iso_time

def capture_at(y, s, top):
    cut = np.quantile(s, 1 - top)
    flagged = s >= cut
    return y[flagged].sum() / y.sum() * 100, y[flagged].mean() * 100

cap_rows = []
for key in [ISO_KEY, ('XGBoost', 'Class weights'), ('Random Forest', 'Class weights'),
            ('Logistic Regression', 'Class weights'), RULE_KEY]:
    r = {'model': f'{key[0]} ({key[1]})', 'PR_AUC': average_precision_score(y_te, scores_te[key]),
         'ROC_AUC': roc_auc_score(y_te, scores_te[key])}
    for top in (0.001, 0.005, 0.01, 0.05):
        r[f'fraud_caught_top_{top:.1%}'], r[f'precision_top_{top:.1%}'] = capture_at(y_te, scores_te[key], top)
    cap_rows.append(r)
anomaly_cmp = pd.DataFrame(cap_rows).round(2)
anomaly_cmp.to_csv(TAB / 'anomaly_vs_supervised.csv', index=False)
print(f'Isolation Forest trained in {iso_time:.1f}s without labels')
anomaly_cmp
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
bins = np.linspace(iso_te.min(), iso_te.max(), 60)
ax[0].hist(iso_te[y_te == 0], bins=bins, color=GOOD, alpha=0.7, density=True, label='genuine')
ax[0].hist(iso_te[y_te == 1], bins=bins, color=BAD, alpha=0.7, density=True, label='fraud')
ax[0].axvline(np.quantile(iso_te, 0.99), color='black', ls='--', label='top-1% cut')
ax[0].set_title('Isolation Forest anomaly score (labels added only for this plot)')
ax[0].set_xlabel('anomaly score')
ax[0].legend()

tops = np.linspace(0.001, 0.05, 50)
for key, c in [(ISO_KEY, '#e09f3e'), (('XGBoost', 'Class weights'), BAD), (('Logistic Regression', 'Class weights'), GOOD)]:
    ax[1].plot(tops * 100, [capture_at(y_te, scores_te[key], t)[0] for t in tops], label=f'{key[0]}', color=c)
ax[1].set_xlabel('% of transactions reviewed (highest scores first)')
ax[1].set_ylabel('% of all fraud caught')
ax[1].set_title('Fraud capture: unsupervised vs supervised')
ax[1].legend()
savefig('05_anomaly_detection')
""")

step('Step 5 summary · Anomaly detection without labels',
     'Trained an Isolation Forest that never sees a fraud label, then checked how much fraud it ranks as most unusual.',
     'Brand-new scams have no labels yet, so a supervised model cannot learn them. We need a safety net that spots "weird" on its own.',
     '300 trees with 256 samples each, fitted on the training features only (`fit(X_tr)` with no `y`). Labels are used only afterwards to score the ranking.',
     'Trained in **1 second**. Its **top 1% most unusual payments hold 50% of all test fraud** (precision 6.5%), and the top 5% hold 84%. The supervised models get 88% in the top 1% (precision 11.4%), so labels roughly double precision. '
     'A lesson in its own right: Isolation Forest has **ROC-AUC 0.95 but PR-AUC only 0.05**. This shows live why ROC-AUC looks optimistic on rare fraud. '
     '**Its role:** a second line of defence and a review queue for unseen patterns, not a replacement for the supervised model.')

# ---------------------------------------------------------------- PS6
md("""
## 6 · Choosing the cut-off by cost (PS6)

Expected cost = **missed frauds × ₹X + wrongly blocked genuine payments × ₹Y**. We slide the cut-off from 0 to 1 on the **validation** slice, pick the cheapest point, and then report that cut-off once on **test**.

**Final model selection criteria** (decided before looking at test):
1. The lowest validation expected cost at its own cost-optimal cut-off.
2. Ties are broken by the higher validation PR-AUC.
""")

code(r"""
cand = comparison[(comparison.threshold_rule == 'cost-optimal') & (comparison.model != 'Rule baseline')]
cand = cand.sort_values(['val_cost_at_opt_₹', 'val_PR_AUC'], ascending=[True, False])
FINAL_KEY = (cand.iloc[0].model, cand.iloc[0].imbalance)
FINAL_THR = best_thr[FINAL_KEY]
print('Selection ranking (validation only):')
display(cand[['model', 'imbalance', 'val_cost_at_opt_₹', 'val_PR_AUC', 'threshold']].round(4).head(9))
print(f'FINAL MODEL: {FINAL_KEY[0]} + {FINAL_KEY[1]}   cost-optimal cut-off = {FINAL_THR:.4f}')
""")

code(r"""
fig, ax = plt.subplots(figsize=(7.5, 5))
plot_keys = [('XGBoost', 'Class weights'), ('XGBoost', 'SMOTE'), ('Random Forest', 'Class weights'),
             ('Logistic Regression', 'Class weights'), ISO_KEY, RULE_KEY]
for key in dict.fromkeys([FINAL_KEY] + plot_keys):
    pr, rc, _ = precision_recall_curve(y_te, scores_te[key])
    lw = 3 if key == FINAL_KEY else 1.4
    ax.plot(rc, pr, lw=lw, label=f'{key[0]} ({key[1]})  PR-AUC={average_precision_score(y_te, scores_te[key]):.3f}')
ax.axhline(y_te.mean(), color='grey', ls=':', label=f'random = {y_te.mean():.4f}')
ax.set_xlabel('Recall (share of fraud caught)')
ax.set_ylabel('Precision (share of alarms that are real)')
ax.set_title('Precision-recall curves (test)')
ax.legend(fontsize=8, loc='upper right')
savefig('06_pr_curves')
""")

code(r"""
s_va, s_te = scores_va[FINAL_KEY], scores_te[FINAL_KEY]
_, grid, costs_va = best_threshold(y_va, s_va)
costs_te = np.array([cost_of(y_te, (s_te >= t).astype(int)) for t in grid])
cost_default_te = cost_of(y_te, (s_te >= 0.5).astype(int))
cost_opt_te = cost_of(y_te, (s_te >= FINAL_THR).astype(int))
cost_nothing_te = y_te.sum() * COST_MISSED_FRAUD

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(grid, costs_va / 1000, label='validation (used to choose)', color=GOOD)
ax.plot(grid, costs_te / 1000, label='test (reported)', color=BAD, alpha=0.7)
ax.axvline(FINAL_THR, color='black', ls='--')
ax.scatter([FINAL_THR], [costs_va.min() / 1000], s=80, color='black', zorder=5,
           label=f'chosen cut-off = {FINAL_THR:.3f}')
ax.axvline(0.5, color='grey', ls=':', label='default 0.5')
ax.set_ylim(0, 2.5 * max(costs_va.min(), costs_te.min()) / 1000)   # zoom past the spike at cut-off ≈ 0 (block everything)
ax.set_xlabel('cut-off (block if score ≥ cut-off)')
ax.set_ylabel('expected cost (₹ thousands)')
ax.set_title(f'Expected cost vs cut-off: {FINAL_KEY[0]} ({FINAL_KEY[1]})')
ax.legend()
savefig('07_cost_vs_threshold')

cost_summary = pd.DataFrame([
    ['Do nothing (approve all)', cost_nothing_te],
    ['Final model @ default 0.5', cost_default_te],
    [f'Final model @ cost-optimal {FINAL_THR:.3f}', cost_opt_te],
], columns=['policy', 'test_expected_cost_₹'])
cost_summary['saving_vs_do_nothing_₹'] = cost_nothing_te - cost_summary['test_expected_cost_₹']
cost_summary
""")

md("""
### 6b · Sensitivity: what if the costs change?

The cut-off depends on the ratio between ₹X (a missed fraud) and ₹Y (a false block). When missing fraud gets more expensive, the best cut-off drops and we block more. The table below re-chooses the cut-off on validation for each cost scenario.
""")

code(r"""
sens_rows = []
base = (COST_MISSED_FRAUD, COST_FALSE_BLOCK)
for X_ in (2_000, 5_000, 10_000, 20_000, 50_000):
    for Y_ in (100, 500, 2_000):
        COST_MISSED_FRAUD, COST_FALSE_BLOCK = X_, Y_
        t, _, _ = best_threshold(y_va, s_va)
        pred = (s_te >= t).astype(int)
        sens_rows.append({'₹X missed fraud': X_, '₹Y false block': Y_, 'X/Y ratio': X_ / Y_, 'cut-off': t,
                          'test_recall_%': recall_score(y_te, pred) * 100,
                          'genuine_blocked': int(((pred == 1) & (y_te == 0)).sum()),
                          'test_cost_₹': cost_of(y_te, pred)})
COST_MISSED_FRAUD, COST_FALSE_BLOCK = base
sens = pd.DataFrame(sens_rows).round(3)
sens.to_csv(TAB / 'cost_sensitivity.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 3.8))
for Y_, g in sens.groupby('₹Y false block'):
    ax.plot(g['₹X missed fraud'], g['cut-off'], marker='o', label=f'₹Y = {Y_}')
ax.set_xscale('log')
ax.set_xlabel('₹X: cost of a missed fraud')
ax.set_ylabel('cost-optimal cut-off')
ax.set_title('Pricier missed fraud means a lower cut-off and more blocking')
ax.legend()
savefig('08_cost_sensitivity')
sens
""")

step('Step 6 summary · Model selection and cost-based cut-off',
     'Picked the final model and its block cut-off by money instead of by habit (0.5).',
     '0.5 is an arbitrary number. The business needs a line it can defend in rupees: each missed fraud costs ₹X and each blocked genuine customer costs ₹Y.',
     'Tried about 1,500 cut-offs on **validation**, where cost = missed × ₹10,000 + false blocks × ₹500. Selection rule, fixed in advance: lowest validation cost, with ties broken by validation PR-AUC. The test set was used once, to report.',
     '**Final model: XGBoost (no extra imbalance handling).** It tied for the lowest validation cost (₹1.30 lakh) with Random Forest and won the tie on PR-AUC (0.786 vs 0.767). It also trains about 14× faster. '
     '**Cut-off 0.21:** on test it catches **58 of 74 frauds (78%)**, blocks only **19 genuine** payments (precision 75%) and costs **₹1.70 lakh**, against ₹1.85 lakh at 0.5 (8% cheaper) and ₹7.4 lakh for doing nothing (77% cheaper). '
     '**Sensitivity:** the cut-off stays at 0.21 for every ₹X/₹Y ratio from 10 to 100. It rises to 0.61 when the ratio is 5 or less, and drops to about 0 when it is 200 or more (then 595 genuine get blocked). So the decision is stable over a wide range of cost assumptions. '
     '*Honest note:* Random Forest with class weights has a slightly higher **test** PR-AUC (0.819), but choosing on test would be cheating, so we keep the validation-based choice.')

# ---------------------------------------------------------------- PS7
md("""
## 7 · The Approve / Step-up OTP / Block rule (PS7)

There are two cut-offs, which make three zones: **score < T1 → approve**, **T1 ≤ score < T2 → step-up OTP**, **score ≥ T2 → block**.

The cost of a zone policy is made of:
- **Fraud approved:** ₹X each.
- **Every OTP challenge:** ₹Z each.
- **Fraud that gets past the OTP:** (1 − OTP stop rate) × ₹X each. *The stop rate is an assumption.*
- **Genuine payment blocked:** ₹Y each.

T1 and T2 are searched over a grid on **validation**, and the results are reported once on **test**.
""")

code(r"""
def zone_cost_grid(y, s, cands):
    fs, gs = np.sort(s[y == 1]), np.sort(s[y == 0])
    F_ge = len(fs) - np.searchsorted(fs, cands, side='left')   # frauds with score >= t
    G_ge = len(gs) - np.searchsorted(gs, cands, side='left')
    T1F, T2F = np.meshgrid(F_ge, F_ge, indexing='ij')
    T1G, T2G = np.meshgrid(G_ge, G_ge, indexing='ij')
    fraud_appr, fraud_otp, gen_otp, gen_block = len(fs) - T1F, T1F - T2F, T1G - T2G, T2G
    cost = (fraud_appr * COST_MISSED_FRAUD + (fraud_otp + gen_otp) * COST_OTP
            + fraud_otp * (1 - OTP_STOP_RATE) * COST_MISSED_FRAUD + gen_block * COST_FALSE_BLOCK).astype(float)
    cost[(fraud_otp + gen_otp) / len(y) > MAX_OTP_SHARE] = np.inf   # friction cap on the OTP zone
    cost[np.triu_indices(len(cands), 1)[::-1]] = np.inf          # require T1 <= T2
    return cost

def choose_zones(y, s):
    # Cheapest (T1, T2) on the given slice; 1.01 lets the search pick "never OTP" / "never block"
    cands = np.unique(np.r_[np.quantile(s, np.r_[np.linspace(0, 0.97, 30), np.linspace(0.97, 1, 600)]), s.max() + 1e-9, 1.01])
    zc = zone_cost_grid(y, s, cands)
    i, j = np.unravel_index(np.argmin(zc), zc.shape)
    return cands[i], cands[j], zc[i, j]

T1, T2, zone_cost_va = choose_zones(y_va, s_va)
print(f'T1 (approve / OTP) = {T1:.4f}   T2 (OTP / block) = {T2:.4f}   validation cost ₹{zone_cost_va:,.0f}')

def band_table(y, s, t1, t2):
    zone = np.where(s >= t2, 'BLOCK', np.where(s >= t1, 'STEP-UP OTP', 'APPROVE'))
    rows = []
    for z in ['APPROVE', 'STEP-UP OTP', 'BLOCK']:
        m = zone == z
        f, g = int(y[m].sum()), int((y[m] == 0).sum())
        cost = {'APPROVE': f * COST_MISSED_FRAUD,
                'STEP-UP OTP': (f + g) * COST_OTP + f * (1 - OTP_STOP_RATE) * COST_MISSED_FRAUD,
                'BLOCK': g * COST_FALSE_BLOCK}[z]
        rows.append({'zone': z, 'transactions': int(m.sum()), '%_of_traffic': m.mean() * 100,
                     'frauds': f, '%_of_all_fraud': f / y.sum() * 100,
                     'genuine_customers_affected': g if z != 'APPROVE' else 0, 'expected_cost_₹': cost})
    t = pd.DataFrame(rows)
    t.loc[len(t)] = ['TOTAL', t.transactions.sum(), 100.0, t.frauds.sum(), 100.0,
                     t.genuine_customers_affected.sum(), t['expected_cost_₹'].sum()]
    return t

bands = band_table(y_te, s_te, T1, T2)
bands.to_csv(TAB / 'risk_bands_test.csv', index=False)
bands.round(3)
""")

md("""
**Honest baselines.** OTP is cheap (₹10) compared with a missed fraud (₹10,000), so challenging payments *at random* already saves money. To show what the **model** adds, the 3-zone rule is compared with two policies that use no model: *OTP everyone*, and *OTP a random 5%* (the same OTP budget, but chosen without a model).
""")

code(r"""
def policy_table(y, s, thr, t1, t2):
    n, F = len(y), y.sum()
    rnd = np.random.default_rng(SEED).random(n)          # "no model": a random ranking
    rows = [
        ['Do nothing (approve all)', F * COST_MISSED_FRAUD],
        ['OTP everyone (no model)', n * COST_OTP + F * (1 - OTP_STOP_RATE) * COST_MISSED_FRAUD],
        [f'OTP a random {MAX_OTP_SHARE:.0%} (no model)',
         band_table(y, rnd, np.quantile(rnd, 1 - MAX_OTP_SHARE), 2)['expected_cost_₹'].iloc[-1]],
        ['Single cut-off @ default 0.5', cost_of(y, (s >= 0.5).astype(int))],
        [f'Single cut-off @ cost-optimal {thr:.3f}', cost_of(y, (s >= thr).astype(int))],
        [f'3-zone rule (T1={t1:.3f}, T2={t2:.3f})', band_table(y, s, t1, t2)['expected_cost_₹'].iloc[-1]],
    ]
    p = pd.DataFrame(rows, columns=['policy', 'test_expected_cost_₹'])
    p['saving_vs_do_nothing_%'] = (1 - p['test_expected_cost_₹'] / rows[0][1]) * 100
    p['cost_per_100k_txns_₹ (scenario estimate)'] = p['test_expected_cost_₹'] * 100_000 / n
    return p

def plot_policies(p, title, name):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(p.policy, p['test_expected_cost_₹'] / 1000,
            color=['grey', '#c8c8c8', '#a0a0a0', '#e09f3e', GOOD, '#2e8b57'])
    for k, v in enumerate(p['test_expected_cost_₹'] / 1000):
        ax.text(v, k, f'  ₹{v:,.0f}k', va='center')
    ax.invert_yaxis()
    ax.set_xlabel('expected cost on test (₹ thousands)')
    ax.set_title(title)
    savefig(name)

policies = policy_table(y_te, s_te, FINAL_THR, T1, T2)
policies.to_csv(TAB / 'policy_costs.csv', index=False)
plot_policies(policies, 'Card / digital channel: which policy loses the least money?', '09_policy_costs')
policies.round(1)
""")

step('Step 7 summary · Approve / OTP / Block rule',
     'Turned the score into 3 actions using two cut-offs (T1 and T2).',
     'With one cut-off you can only allow or block. Many payments are only "a bit suspicious", and a 5-second OTP is a cheap middle path: easy for the real customer, hard for someone using a stolen account.',
     'A vectorised grid search over about 600 × 600 (T1, T2) pairs on **validation**. Cost = approved fraud × ₹X + OTPs × ₹Z + fraud that gets past the OTP (20%, an *assumption*) × ₹X + blocked genuine × ₹Y. Reported once on test.',
     '**On test:** Approve takes **96.7%** of traffic and lets 10 frauds through. OTP takes **3.2%** (1,836 payments): it catches 8 frauds and asks 1,828 genuine users for an OTP. Block takes **0.11%** (65 payments) and catches **56 frauds (76%)** while wrongly blocking **only 9 genuine** customers. '
     '**Total cost ₹1.39 lakh:** 18% cheaper than the single cut-off (₹1.70 lakh) and **81% cheaper than doing nothing** (₹7.4 lakh). Scaled up, that is about **₹2.4 lakh vs ₹13 lakh per 100k transactions** *(a scenario estimate under the placeholder costs)*.')

# ---------------------------------------------------------------- PS8
md("""
## 8 · Final model: evidence and explanation (PS8)
""")

code(r"""
final_pred = (s_te >= FINAL_THR).astype(int)
trap_final = pd.DataFrame({
    'model': ['Do-nothing', f'{FINAL_KEY[0]} ({FINAL_KEY[1]})'],
    'accuracy_%': [accuracy_score(y_te, np.zeros_like(y_te)) * 100, accuracy_score(y_te, final_pred) * 100],
    'recall_%': [0.0, recall_score(y_te, final_pred) * 100],
    'precision_%': [0.0, precision_score(y_te, final_pred) * 100],
})
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
trap_final.set_index('model')[['accuracy_%', 'recall_%', 'precision_%']].T.plot.bar(ax=ax[0], color=['grey', BAD], rot=0)
ax[0].set_title('The accuracy trap: nearly equal accuracy, completely different fraud caught')
ax[0].set_ylim(0, 105)
for c in ax[0].containers:
    ax[0].bar_label(c, fmt='%.1f', fontsize=8)

cm = confusion_matrix(y_te, final_pred)
money = np.array([[0, cm[0, 1] * COST_FALSE_BLOCK], [cm[1, 0] * COST_MISSED_FRAUD, 0]])
labels = np.array([[f'{cm[i, j]:,}\n₹{money[i, j]:,.0f}' for j in range(2)] for i in range(2)])
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax[1],
            xticklabels=['pred genuine', 'pred fraud'], yticklabels=['actual genuine', 'actual fraud'])
ax[1].set_title(f'Confusion matrix @ cut-off {FINAL_THR:.3f} (test)')
savefig('10_accuracy_trap_and_confusion')
trap_final.round(3)
""")

code(r"""
def explain(model, X_background, X_sample):
    # SHAP values for any of our 9 models (linear or tree, with or without a pipeline)
    est = model.named_steps['m'] if isinstance(model, Pipeline) else model
    if isinstance(est, LogisticRegression):
        sc = model.named_steps['sc']
        sv = shap.LinearExplainer(est, sc.transform(X_background))(sc.transform(X_sample))
        sv.feature_names = list(X_sample.columns)
    else:
        sv = shap.TreeExplainer(est)(X_sample)
        if sv.values.ndim == 3:        # random forest returns one slice per class
            sv = sv[:, :, 1]
    return sv

final_model = MODELS.get(FINAL_KEY)
sv = explain(final_model, X_tr, X_te.sample(3000, random_state=SEED))
plt.figure()
shap.plots.beeswarm(sv, max_display=12, show=False)
plt.title(f'SHAP: what pushes {FINAL_KEY[0]} towards "fraud"')
savefig('11_shap_summary')

importance = pd.DataFrame({'feature': FEATURES, 'mean_|SHAP|': np.abs(sv.values).mean(0)}) \
    .sort_values('mean_|SHAP|', ascending=False).reset_index(drop=True)
importance.to_csv(TAB / 'feature_importance_shap.csv', index=False)
importance.head(10).round(4)
""")

code(r"""
# Save everything the demo app and the report need
joblib.dump({'model': final_model, 'features': FEATURES, 'key': FINAL_KEY,
             'threshold': FINAL_THR, 'T1': T1, 'T2': T2, 'iso': iso}, MOD / 'final_model.joblib')
test.assign(score=s_te, anomaly=iso_te).drop(columns=['amount_band'], errors='ignore') \
    .sample(2000, random_state=SEED).to_csv(MOD / 'demo_transactions.csv', index=False)

fin = comparison[(comparison.model == FINAL_KEY[0]) & (comparison.imbalance == FINAL_KEY[1])
                 & (comparison.threshold_rule == 'cost-optimal')].iloc[0]
iso_row = anomaly_cmp[anomaly_cmp.model.str.startswith('Isolation')].iloc[0]
summary = {
    'costs': {'missed_fraud': COST_MISSED_FRAUD, 'false_block': COST_FALSE_BLOCK, 'otp': COST_OTP,
              'otp_stop_rate': OTP_STOP_RATE, 'note': 'PLACEHOLDER: replace with organiser values'},
    'data': {'cc_rows_clean': len(cc), 'cc_frauds': int(cc.Class.sum()), 'cc_duplicates_dropped': int(dup_cc.sum()),
             'upi_rows': len(upi), 'upi_frauds': int(upi.fraud_flag.sum()),
             'upi_best_test_roc_auc': float(upi_signal.test_ROC_AUC.max())},
    'trap': {'dummy_accuracy_%': float(trap['accuracy_%'][0]), 'dummy_recall_%': 0.0},
    'final_model': {'name': f'{FINAL_KEY[0]} ({FINAL_KEY[1]})', 'threshold': float(FINAL_THR),
                    'test_PR_AUC': float(fin.PR_AUC), 'test_ROC_AUC': float(fin.ROC_AUC),
                    'test_precision': float(fin.precision), 'test_recall': float(fin.recall), 'test_F1': float(fin.F1),
                    'test_TP': int(fin.TP), 'test_FP': int(fin.FP), 'test_FN': int(fin.FN)},
    'anomaly': {'test_PR_AUC': float(iso_row.PR_AUC), 'fraud_in_top_1%': float(iso_row['fraud_caught_top_1.0%'])},
    'bands': {'T1': float(T1), 'T2': float(T2), 'table': bands.round(3).to_dict('records')},
    'policies': policies.round(1).to_dict('records'),
    'top_features': importance.feature.head(5).tolist(),
}
(ROOT / 'outputs' / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str)[:3000])
""")

step('Step 8 summary · Explaining the final model',
     'Explained which inputs push the XGBoost score up or down, and saved the model for the demo app.',
     'Management and the judges must be able to trust *why* a payment is blocked, and an unexplained black box is hard to defend in Q&A.',
     'SHAP `TreeExplainer` on 3,000 random test rows, plus the mean |SHAP| ranking. The model, cut-offs and Isolation Forest are saved to `models/final_model.joblib`.',
     'Top drivers: **V4, V14, V12, V10, V11, then hour of day and amount**. Time and amount still matter even next to the strong PCA features, which matches the EDA. Low V14 / V12 / V10 and high V4 push the score towards fraud. '
     '*Limitation:* the V-columns are anonymised, so we can rank them but not name what they mean in business terms.')

# ---------------------------------------------------------------- UPI CHANNEL
md("""
## 9 · UPI channel: the same pipeline from start to finish

Sections 4-8 built the system on the **card / digital-payment** channel. Here we run the **exact same pipeline** on the **UPI** channel:
- the same 9 models, a rule baseline and an Isolation Forest;
- the same cost-based cut-off, the same selection rule and the same 3-zone search with the 5% OTP cap.

Nothing is tuned specially for UPI, so the two channels are compared fairly.

- **Features:** the one-hot UPI columns from Section 3b. `transaction id` and `transaction_status` (leakage) are excluded, and so is the raw timestamp.
- **Split:** time-aware 60 / 20 / 20, the same as for cards.
- **SMOTE note:** SMOTE blends neighbouring rows, so on one-hot columns it creates "in-between" categories. That's acceptable as a comparison point, and class weights are the cleaner method here.
""")

code(r"""
Xu_tr, Xu_va, Xu_te = upi_X.iloc[:u1], upi_X.iloc[u1:u2], upi_X.iloc[u2:]
yu_tr, yu_va, yu_te = upi_y[:u1], upi_y[u1:u2], upi_y[u2:]
print(f'UPI split  train {len(yu_tr):,} ({yu_tr.sum()} fraud) | validation {len(yu_va):,} ({yu_va.sum()}) | '
      f'test {len(yu_te):,} ({yu_te.sum()})   features: {upi_X.shape[1]}')

UPI_MODELS = make_models((yu_tr == 0).sum() / yu_tr.sum())
u_va, u_te, u_fit = {}, {}, {}
for key, model in UPI_MODELS.items():
    t0 = time.time()
    model.fit(Xu_tr, yu_tr)
    u_fit[key] = time.time() - t0
    u_va[key] = model.predict_proba(Xu_va)[:, 1]
    u_te[key] = model.predict_proba(Xu_te)[:, 1]
    print(f'{key[0]:<20} {key[1]:<14} fit {u_fit[key]:6.1f}s  val PR-AUC {average_precision_score(yu_va, u_va[key]):.4f}')

def upi_rule(df):
    # The only bumps the EDA showed: big amounts (> ₹5,000) and late night (0-3 AM)
    return ((df['amount (INR)'] > 5000).astype(int) + df.hour_of_day.between(0, 3).astype(int)) / 2

u_va[RULE_KEY], u_te[RULE_KEY] = upi_rule(upi.iloc[u1:u2]).values, upi_rule(upi.iloc[u2:]).values
t0 = time.time()
upi_iso = IsolationForest(n_estimators=300, max_samples=256, random_state=SEED, n_jobs=-1).fit(Xu_tr)  # no y!
u_fit[ISO_KEY], u_fit[RULE_KEY] = time.time() - t0, 0.0
u_va[ISO_KEY], u_te[ISO_KEY] = -upi_iso.score_samples(Xu_va), -upi_iso.score_samples(Xu_te)

rows = []
for key in u_te:
    t, _, _ = best_threshold(yu_va, u_va[key])
    rows.append({'model': key[0], 'imbalance': key[1], 'cut_off': t,
                 'val_PR_AUC': average_precision_score(yu_va, u_va[key]),
                 'val_cost_at_opt_₹': cost_of(yu_va, (u_va[key] >= t).astype(int)),
                 'fraud_in_top_1%_%': capture_at(yu_te, u_te[key], 0.01)[0],
                 'fit_s': u_fit[key], **evaluate(yu_te, u_te[key], t)})
upi_cmp = pd.DataFrame(rows)
upi_cmp['PR_AUC_lift_vs_random'] = upi_cmp.PR_AUC / yu_te.mean()
upi_cmp.to_csv(TAB / 'upi_model_comparison.csv', index=False)
print(f'random-guess PR-AUC on UPI test = fraud rate = {yu_te.mean():.4f}')
upi_cmp[['model', 'imbalance', 'val_PR_AUC', 'PR_AUC', 'PR_AUC_lift_vs_random', 'ROC_AUC', 'fraud_in_top_1%_%',
         'cut_off', 'TP', 'FP', 'FN', 'expected_cost_₹', 'fit_s']].round(4)
""")

code(r"""
# Same selection rule as the card channel: lowest validation cost, ties broken by validation PR-AUC
ucand = upi_cmp[~upi_cmp.model.isin(['Rule baseline', 'Isolation Forest'])] \
    .sort_values(['val_cost_at_opt_₹', 'val_PR_AUC'], ascending=[True, False])
UPI_FINAL = (ucand.iloc[0].model, ucand.iloc[0].imbalance)
UPI_THR = ucand.iloc[0].cut_off
su_va, su_te = u_va[UPI_FINAL], u_te[UPI_FINAL]
uT1, uT2, _ = choose_zones(yu_va, su_va)
print(f'UPI final model: {UPI_FINAL[0]} ({UPI_FINAL[1]})   single cut-off {UPI_THR:.4f}   T1 {uT1:.4f}   T2 {uT2:.4f}')

upi_bands = band_table(yu_te, su_te, uT1, uT2)
upi_bands.to_csv(TAB / 'upi_risk_bands_test.csv', index=False)
display(upi_bands.round(3))

upi_policies = policy_table(yu_te, su_te, UPI_THR, uT1, uT2)
upi_policies.to_csv(TAB / 'upi_policy_costs.csv', index=False)
plot_policies(upi_policies, 'UPI channel: which policy loses the least money?', '13_upi_policy_costs')
upi_policies.round(1)
""")

code(r"""
# 500 rows: exact SHAP on a deep Random Forest with 70 one-hot columns is slow (3,000 rows took > 8 min)
sv_u = explain(UPI_MODELS[UPI_FINAL], Xu_tr, Xu_te.sample(500, random_state=SEED))
upi_importance = pd.DataFrame({'feature': upi_X.columns, 'mean_|SHAP|': np.abs(sv_u.values).mean(0)}) \
    .sort_values('mean_|SHAP|', ascending=False).reset_index(drop=True)
upi_importance.to_csv(TAB / 'upi_feature_importance_shap.csv', index=False)
plt.figure()
shap.plots.beeswarm(sv_u, max_display=10, show=False)
plt.title(f'UPI SHAP: {UPI_FINAL[0]} ({UPI_FINAL[1]})')
savefig('14_upi_shap_summary')
upi_importance.head(8).round(4)
""")

step('Step 9 summary · UPI channel, the full pipeline',
     'Ran the *identical* pipeline on the UPI payments: 9 models, the rule baseline, the Isolation Forest, the cost-based cut-off, the selection rule, the 3 zones and SHAP.',
     'The challenge is "UPI / digital payment" fraud, so the UPI channel gets the same full treatment, not just a quick check. Using the same pipeline makes the comparison with the card channel fair.',
     'The same code with no UPI-specific tuning: a time-aware 60/20/20 split (50,000 test payments with 96 frauds), the leakage columns excluded, and one-hot features. Choices were made on validation and reported on test.',
     '**Every model is about as good as random:** test PR-AUC 0.0021-0.0025 against a random 0.0019 (only 1.1-1.3× better) and ROC-AUC 0.51-0.54. The Isolation Forest is no better. '
     '**The cost rule correctly refuses to block:** at each model\'s cost cut-off, 0 frauds are caught and only 0-3 payments are blocked, so the system does **not** annoy customers on a signal it doesn\'t have. All models tie on validation cost (₹9.6 lakh, "block nothing"), and Random Forest (SMOTE) wins the tie on PR-AUC. '
     '**3 zones:** approve 97.4%, OTP 2.6%, block 2 payments. That is a 1.9% saving, which is noise (4 frauds in the OTP zone vs about 2.5 expected by chance). '
     '**SHAP agrees:** the biggest UPI feature effect (mean |SHAP| 0.015) is about **100× smaller** than on cards (1.58), and the "top" features are scattered (an age group, a merchant category, one bank). That pattern is noise, not a fraud signal. '
     '**A trap we avoided:** "OTP everyone" would look 28% cheaper, but only because OTP (₹10) is so cheap next to a missed fraud (₹10,000), and it challenges 100% of customers. The 5% friction cap rules it out.')

md("""
### 9b · Can the UPI data be "cleaned" into good results? (the shuffled-label test)

There are two possible reasons the UPI results are low: **(a) we didn't try hard enough**, or **(b) there is no pattern to find**. To tell them apart:
1. We add **14 engineered features**: round amounts, last digit, same bank, age gap, minute and second, night flag, amount vs its category's typical value, and the transaction-ID order.
2. We train a strong model (gradient boosting, 5-fold cross-validation).
3. We then **shuffle the fraud labels at random 20 times** and repeat. Shuffled labels have *no* pattern by construction. If the real labels hold a pattern, they must beat the shuffled ones clearly.
""")

code(r"""
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
from upi_rescue_test import rescue_test

t0 = time.time()
rescue = rescue_test(upi_raw, n_shuffles=20, seed=SEED)
print(f"{rescue['n_features']} features ({rescue['n_engineered']} engineered) | {time.time() - t0:.0f}s")
print(f"REAL labels      ROC-AUC {rescue['real_roc_auc']:.4f}   PR-AUC {rescue['real_pr_auc']:.5f}")
print(f"SHUFFLED labels  ROC-AUC {rescue['shuffled_roc_auc'].mean():.4f} ± {rescue['shuffled_roc_auc'].std():.4f}"
      f"   PR-AUC {rescue['shuffled_pr_auc'].mean():.5f}")
print(f"p-value = {rescue['p_value']:.3f}  (share of shuffles that do at least as well as the real labels)")

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(rescue['shuffled_roc_auc'], bins=10, color='#c8c8c8', edgecolor='white', label='20 × shuffled (no-pattern) labels')
ax.axvline(rescue['real_roc_auc'], color=BAD, lw=3, label=f"real UPI labels = {rescue['real_roc_auc']:.3f}")
ax.axvline(0.5, color='black', ls=':', label='coin flip = 0.5')
ax.set_xlabel('ROC-AUC (5-fold, 27 features)')
ax.set_ylabel('count')
ax.set_title(f"UPI: real labels look exactly like shuffled ones (p = {rescue['p_value']:.2f})")
ax.legend(fontsize=8)
savefig('15_upi_shuffled_label_test')
pd.DataFrame({'run': ['real'] + [f'shuffle_{i}' for i in range(1, 21)],
              'roc_auc': np.r_[rescue['real_roc_auc'], rescue['shuffled_roc_auc']],
              'pr_auc': np.r_[rescue['real_pr_auc'], rescue['shuffled_pr_auc']]}).to_csv(TAB / 'upi_shuffled_label_test.csv', index=False)
""")

step('Step 9b summary · The shuffled-label test',
     'Tried every fair way to get signal out of the UPI data (14 extra features and a strong model), then compared the real labels with labels we shuffled ourselves.',
     'To prove the low UPI results are caused by the *data*, not by weak modelling, and to answer "why don\'t you just clean it?"',
     '`scripts/upi_rescue_test.py`: gradient boosting, 5-fold cross-validation, 27 features, then 20 label shuffles as a "no pattern" reference.',
     '**Real labels: ROC-AUC 0.499. Shuffled labels: 0.495 ± 0.013. p = 0.43.** The real fraud flags score no better than random labels, so they were assigned without regard to any column. '
     '**Cleaning can\'t help:** cleaning fixes broken *values*, and the file has none. The problem is that the answer column doesn\'t depend on the inputs. '
     '"Fixing" it would mean inventing our own fraud labels, which is circular and counts as fabrication under the rules.')

md("""
## 10 · Two channels side by side

This is the most important comparison in the project. The **same pipeline**, applied to two channels, gives very different results, and the difference comes from the **data**, not the method.
""")

code(r"""
def channel_row(name, y, s_te_dict, final_key, thr, t1, t2, pol):
    s = s_te_dict[final_key]
    pred = (s >= thr).astype(int)
    pol = pol.set_index('policy')['test_expected_cost_₹']
    return {'channel': name, 'test_transactions': len(y), 'test_frauds': int(y.sum()),
            'fraud_rate_%': y.mean() * 100, 'final_model': f'{final_key[0]} ({final_key[1]})',
            'PR_AUC': average_precision_score(y, s), 'random_PR_AUC': y.mean(),
            'PR_AUC_lift_x': average_precision_score(y, s) / y.mean(), 'ROC_AUC': roc_auc_score(y, s),
            'fraud_in_top_1%_%': capture_at(y, s, 0.01)[0],
            'iso_forest_top_1%_%': capture_at(y, s_te_dict[ISO_KEY], 0.01)[0],
            'recall_at_cut_off_%': recall_score(y, pred) * 100, 'genuine_blocked': int(((pred == 1) & (y == 0)).sum()),
            '3zone_saving_vs_nothing_%': (1 - pol.iloc[-1] / pol.iloc[0]) * 100,
            '3zone_saving_vs_random_OTP_%': (1 - pol.iloc[-1] / pol.iloc[2]) * 100}

channels = pd.DataFrame([
    channel_row('Card / digital', y_te, scores_te, FINAL_KEY, FINAL_THR, T1, T2, policies),
    channel_row('UPI', yu_te, u_te, UPI_FINAL, UPI_THR, uT1, uT2, upi_policies),
])
channels.to_csv(TAB / 'two_channel_comparison.csv', index=False)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for a, (title, y, sd, fk) in zip(ax[:2], [('Card / digital channel', y_te, scores_te, FINAL_KEY),
                                          ('UPI channel', yu_te, u_te, UPI_FINAL)]):
    for key in dict.fromkeys([fk, ('Logistic Regression', 'Class weights'), ('Random Forest', 'Class weights'), ISO_KEY]):
        pr, rc, _ = precision_recall_curve(y, sd[key])
        a.plot(rc, pr, lw=3 if key == fk else 1.3, label=f'{key[0]} ({key[1]})  PR-AUC={average_precision_score(y, sd[key]):.3f}')
    a.axhline(y.mean(), color='grey', ls=':', label=f'random = {y.mean():.4f}')
    a.set_xlabel('recall')
    a.set_ylabel('precision')
    a.set_title(f'{title}: precision-recall (test)')
    a.legend(fontsize=8)
ax[2].bar(channels.channel, channels.PR_AUC_lift_x, color=[GOOD, BAD])
ax[2].axhline(1, color='grey', ls=':', label='1× = random guessing')
for k, v in enumerate(channels.PR_AUC_lift_x):
    ax[2].text(k, v, f'{v:,.1f}×', ha='center', va='bottom')
ax[2].set_yscale('log')
ax[2].set_title('How many times better than random? (PR-AUC ÷ fraud rate)')
ax[2].legend()
savefig('12_two_channel_comparison')
channels.round(3).T
""")

code(r"""
summary['upi_channel'] = {
    'final_model': f'{UPI_FINAL[0]} ({UPI_FINAL[1]})', 'threshold': float(UPI_THR), 'T1': float(uT1), 'T2': float(uT2),
    'comparison': upi_cmp.round(4).to_dict('records'), 'bands': upi_bands.round(3).to_dict('records'),
    'policies': upi_policies.round(1).to_dict('records'), 'top_features': upi_importance.feature.head(5).tolist(),
}
summary['two_channels'] = channels.round(4).to_dict('records')

# UPI bundle for the demo app: model, cut-offs, Isolation Forest, and 2,000 test payments (raw + model features)
demo_idx = Xu_te.sample(2000, random_state=SEED).index
joblib.dump({'model': UPI_MODELS[UPI_FINAL], 'features': list(upi_X.columns), 'key': UPI_FINAL,
             'threshold': UPI_THR, 'T1': uT1, 'T2': uT2, 'iso': upi_iso,
             'demo_X': Xu_te.loc[demo_idx], 'demo_raw': upi.loc[demo_idx, upi_raw.columns]}, MOD / 'upi_model.joblib')
(ROOT / 'outputs' / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
print('summary.json updated with the UPI channel')
""")

step('Step 10 summary · Two channels side by side',
     'Put the two channels next to each other: the same pipeline, two datasets.',
     'To show the judges that the *method* works and that the difference between the channels comes from the *data*.',
     'One comparison table and one chart: PR curves for each channel, plus "how many times better than random" (PR-AUC ÷ fraud rate) on a log scale.',
     '**Card / digital:** the model is about **600× better than random**, catches **78%** of fraud and saves **81%** vs doing nothing. '
     '**UPI:** only **about 1.2× random**, catches 0%, and saves about 2% (noise). '
     '**Conclusion:** the pipeline learns real fraud patterns when the labels carry them, *and it recognises when they don\'t*, so it stays safe instead of blocking customers at random. The UPI gap is a **labelling problem, not a model problem** (recommendation #3).')

md("""
## Three findings → three recommendations

| # | Finding (observed) | Evidence | Recommendation (action) | Priority |
|---|---|---|---|---|
| 1 | A cost-tuned 3-zone rule loses far less money than a single 0.5 cut-off or doing nothing | Test cost ₹1.39 lakh vs ₹1.85 lakh (0.5 cut-off) vs ₹7.4 lakh (nothing). It blocks only 0.11% of traffic and challenges 3.2% | **Deploy XGBoost with the Approve / OTP / Block bands** as a pilot on a slice of live traffic, and re-tune T1 and T2 every month with the official ₹ costs | 🔴 High |
| 2 | A detector that never saw a label still finds half of all fraud in its top 1% | Isolation Forest: 50% of fraud in the top 1%, 84% in the top 5%, trained in 1 second | **Run the anomaly score as a second line of defence**: send the top 1% of unusual payments that the main model approved to a human review queue, to catch new scam types early | 🟠 Medium |
| 3 | On the UPI channel, the same pipeline finds no learnable fraud signal | 9 models + the Isolation Forest are all 1.1-1.3× random (ROC-AUC 0.51-0.54), vs about 600× on cards; no UPI column survives the Bonferroni correction | **Audit how UPI fraud is labelled** (who flags it, when, and how it is confirmed). Until that's fixed, run UPI on approve + a capped OTP and **do not auto-block** | 🔴 High |

Supporting pattern for #1: fraud is about 5× more common between 1 and 5 AM, and is over-represented in ₹0-1 "card-testing" amounts. These are associations, so use them as inputs to the model and the OTP trigger, not as stand-alone block rules. The pure rule baseline cost ₹36 lakh.

> All savings are **scenario estimates** under the placeholder costs. They are not observed results.
""")

md("""
## Limitations

- **The costs are placeholders.** ₹X, ₹Y and ₹Z must be replaced with the organiser values. The cut-offs move with them (see 6b).
- **The OTP stop rate (80%) is an assumption.** It can't be measured from this data.
- **creditcard.csv covers only about 2 days.** Seasonal and weekly patterns can't be tested, and the test slice holds relatively few frauds, so the test metrics carry real sampling noise.
- **V1 to V28 are anonymised PCA components.** Explanations can rank them, but not name what they mean in business terms.
- **The UPI dataset is simulated** (as the brief itself says), and its labels show no learnable signal. UPI results show how the *pipeline behaves on weak data*; they don't show UPI fraud patterns.
- **SMOTE on one-hot UPI columns** creates "in-between" categories. Class weights are the cleaner method there (SMOTENC would be the proper fix).
- **The 5% OTP friction cap is our assumption.** A different cap would move T1.
- All patterns are **observational associations**. All savings figures are **scenario estimates** based on the placeholder costs.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, OUT)
print('wrote', OUT, len(cells), 'cells')
