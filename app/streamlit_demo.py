"""Fraud decision demo: pick a channel and a transaction, and see its risk score and decision (Approve / OTP / Block).

Run from the project root:  .venv/bin/streamlit run app/streamlit_demo.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource
def load(name):
    # One channel at a time, so a missing UPI model never takes the card demo down with it
    bundle = joblib.load(ROOT / 'models' / name)
    if name == 'final_model.joblib':
        bundle['demo'] = pd.read_csv(ROOT / 'models' / 'demo_transactions.csv')
    return bundle


def load_or_explain(name):
    try:
        return load(name)
    except FileNotFoundError:
        st.error(f'`models/{name}` is missing. Run the notebook first (it creates the model files):\n\n'
                 '`cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace fraud_analysis.ipynb`')
    except ModuleNotFoundError as e:
        st.error(f'This Python is missing the `{e.name}` package. Start the demo with the project environment:\n\n'
                 '`.venv/bin/streamlit run app/streamlit_demo.py`')
    st.stop()


def card_features(df, names):
    X = df[[f'V{i}' for i in range(1, 29)]].copy()
    X['log_amount'] = np.log1p(df.Amount)
    X['hour_sin'] = np.sin(2 * np.pi * df.hour / 24)
    X['hour_cos'] = np.cos(2 * np.pi * df.hour / 24)
    return X[names]


def decide(score, t1, t2):
    if score >= t2:
        return '⛔ BLOCK', 'High risk: stop the payment and alert the customer.'
    if score >= t1:
        return '📱 STEP-UP OTP', 'Doubtful: ask for an OTP before approving.'
    return '✅ APPROVE', 'Looks normal: approve with no friction.'


def pick_row(labels, key):
    pick = st.radio('Pick a transaction', ['Random genuine', 'Random fraud', 'By row number'], horizontal=True, key=f'pick_{key}')
    if pick == 'By row number':
        return int(st.number_input('Row', 0, len(labels) - 1, 0, key=f'row_{key}'))
    pool = np.flatnonzero(labels == (1 if pick == 'Random fraud' else 0))
    state = f'idx_{key}'
    if st.button('🎲 Draw another', key=f'draw_{key}') or state not in st.session_state \
            or st.session_state.get(f'pool_{key}') != pick:
        st.session_state[state] = int(np.random.choice(pool))
        st.session_state[f'pool_{key}'] = pick
    return st.session_state[state]


def show(score, anomaly, actual, t1, t2):
    label, why = decide(score, t1, t2)
    st.metric('Fraud risk score', f'{score:.4f}')
    st.subheader(label)
    st.write(why)
    st.progress(min(score, 1.0))
    st.write(f'Anomaly score (Isolation Forest, trained without labels): **{anomaly:.3f}**')
    st.write(f"Actual label in data: **{'FRAUD' if actual == 1 else 'genuine'}**")


st.set_page_config(page_title='Fraud Decision Demo', page_icon='🛡️')
st.title('🛡️ Payment Fraud Decision Engine')
channel = st.radio('Channel', ['Card / digital', 'UPI'], horizontal=True)

if channel == 'Card / digital':
    card = load_or_explain('final_model.joblib')
    st.caption(f"Model: {card['key'][0]} ({card['key'][1]}) · T1 = {card['T1']:.4f} · T2 = {card['T2']:.3f}")
    demo = card['demo']
    row = demo.iloc[[pick_row(demo.Class.values, 'card')]].copy()
    c1, c2 = st.columns(2)
    row['Amount'] = c1.number_input('Amount (edit to test)', 0.0, 30000.0, float(row.Amount.iloc[0]))
    row['hour'] = c2.slider('Hour of day (edit to test)', 0, 23, int(row.hour.iloc[0]))
    X = card_features(row, card['features'])
    show(float(card['model'].predict_proba(X)[:, 1][0]), float(-card['iso'].score_samples(X)[0]),
         int(row.Class.iloc[0]), card['T1'], card['T2'])
    with st.expander('Raw features'):
        st.dataframe(row.T, width='stretch')
else:
    upi = load_or_explain('upi_model.joblib')
    st.caption(f"Model: {upi['key'][0]} ({upi['key'][1]}) · T1 = {upi['T1']:.4f} · T2 = {upi['T2']:.3f}")
    st.info('On UPI the labels carry no learnable pattern (real labels score the same as shuffled ones, p = 0.43). '
            'Watch how frauds and genuine payments get similar scores, so the system rarely blocks.')
    raw, Xd = upi['demo_raw'], upi['demo_X']
    i = pick_row(raw.fraud_flag.values, 'upi')
    r = raw.iloc[i]
    st.write(f"**{r['transaction type']}** · {r['merchant_category']} · ₹{r['amount (INR)']:,} · "
             f"{r['sender_bank']} → {r['receiver_bank']} · {r['device_type']} / {r['network_type']} · "
             f"{pd.Timestamp(r['timestamp']):%d %b %Y, %H:%M}")
    X = Xd.iloc[[i]][upi['features']]
    show(float(upi['model'].predict_proba(X)[:, 1][0]), float(-upi['iso'].score_samples(X)[0]),
         int(r.fraud_flag), upi['T1'], upi['T2'])
    with st.expander('Raw transaction'):
        st.dataframe(raw.iloc[[i]].T.astype(str), width='stretch')   # mixed column types: show as text
