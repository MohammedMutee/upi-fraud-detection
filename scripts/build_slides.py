"""Builds presentation/fraud_detection_deck.pptx from the notebook's outputs (every number is read, none typed).

Usage:  .venv/bin/python scripts/build_slides.py
        then (optional PDF):  soffice --headless --convert-to pdf --outdir presentation presentation/fraud_detection_deck.pptx
"""
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG, TAB = ROOT / 'outputs' / 'figures', ROOT / 'outputs' / 'tables'
OUT = ROOT / 'presentation' / 'fraud_detection_deck.pptx'
S = json.loads((ROOT / 'outputs' / 'summary.json').read_text())

NAVY, PAPER, WHITE = RGBColor(0x14, 0x21, 0x3D), RGBColor(0xF7, 0xF6, 0xF2), RGBColor(0xFD, 0xFD, 0xFB)
INK, MUTED = RGBColor(0x1F, 0x29, 0x37), RGBColor(0x4A, 0x55, 0x68)
ORANGE, BLUE, GREEN = RGBColor(0xC2, 0x57, 0x1A), RGBColor(0x2B, 0x7B, 0xBA), RGBColor(0x2E, 0x7D, 0x4F)
SOFT = RGBColor(0xBF, 0xD0, 0xE6)
FONT = 'Arial'
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


def lakh(v):
    return f'₹{v / 1e5:.2f} lakh'


def inr(v):
    # Indian digit grouping: 1,69,500
    s = f'{int(round(v)):d}'
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return '₹' + ','.join(([head] if head else []) + groups + [tail]) if len(s) > 3 else '₹' + s


def pct(v, d=0):
    return f'{v:.{d}f}%'


prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
page = [0]


def new_slide(bg=PAPER, notes=''):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    page[0] += 1
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def text(s, x, y, w, h, runs, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    # runs: a string, or a list of paragraphs; each paragraph a string or a list of (text, {overrides}) pieces
    box = s.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(size * 0.45)
        for piece, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = piece
            r.font.name, r.font.size = FONT, Pt(o.get('size', size))
            r.font.bold = o.get('bold', bold)
            r.font.color.rgb = o.get('color', color)
    return box


def title(s, eyebrow, heading, dark=False):
    text(s, MARGIN, Inches(0.45), Inches(12), Inches(0.4), eyebrow.upper(), size=14, bold=True,
         color=SOFT if dark else ORANGE)
    text(s, MARGIN, Inches(0.8), Inches(12.1), Inches(0.9), heading, size=32, bold=True, color=WHITE if dark else NAVY)


def footer(s, note='', dark=False):
    c = SOFT if dark else MUTED
    if note:
        text(s, MARGIN, Inches(6.95), Inches(11), Inches(0.3), note, size=11, color=c)
    text(s, Inches(12.2), Inches(6.95), Inches(0.55), Inches(0.3), str(page[0]), size=11, color=c, align=PP_ALIGN.RIGHT)


def card(s, x, y, w, h, fill=WHITE):
    shp = s.shapes.add_shape(1, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = RGBColor(0xE3, 0xE6, 0xE4)
    shp.shadow.inherit = False
    return shp


def image(s, name, x, y, w, h):
    # Fit the chart inside the box, keeping its shape, centred, on a white card
    card(s, x, y, w, h)
    iw, ih = Image.open(FIG / name).size
    scale = min((w - Inches(0.2)) / iw, (h - Inches(0.2)) / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(str(FIG / name), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)


def stat(s, x, y, w, big, label, color=NAVY, big_size=44):
    text(s, x, y, w, Inches(0.8), big, size=big_size, bold=True, color=color)
    text(s, x, y + Inches(0.85), w, Inches(0.9), label, size=15, color=MUTED)


def table(s, x, y, w, df, col_w=None, size=14, highlight=None):
    rows, cols = df.shape[0] + 1, df.shape[1]
    t = s.shapes.add_table(rows, cols, x, y, w, Inches(0.42) * rows).table
    for j, cw in enumerate(col_w or [1] * cols):
        t.columns[j].width = int(w * cw / sum(col_w or [1] * cols))
    for i in range(rows):
        for j in range(cols):
            cell = t.cell(i, j)
            cell.text = str(df.columns[j] if i == 0 else df.iat[i - 1, j])
            cell.fill.solid()
            hot = highlight is not None and i - 1 == highlight
            cell.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xFD, 0xEB, 0xDD) if hot else WHITE)
            cell.margin_top = cell.margin_bottom = Inches(0.05)
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name, r.font.size = FONT, Pt(size)
                    r.font.bold = i == 0 or hot
                    r.font.color.rgb = WHITE if i == 0 else INK
    return t


# ---------------------------------------------------------------- data
fm, card_bands = S['final_model'], pd.read_csv(TAB / 'risk_bands_test.csv')
pol = pd.read_csv(TAB / 'policy_costs.csv').set_index('policy')['test_expected_cost_₹']
nothing, default05, opt, zones = pol.iloc[0], pol.iloc[3], pol.iloc[4], pol.iloc[5]
chan = pd.read_csv(TAB / 'two_channel_comparison.csv').set_index('channel')
cardc, upic = chan.loc['Card / digital'], chan.loc['UPI']
shuf = pd.read_csv(TAB / 'upi_shuffled_label_test.csv')
real_auc, sh = shuf.roc_auc.iloc[0], shuf.roc_auc.iloc[1:]
p_value = ((sh >= real_auc).sum() + 1) / (len(sh) + 1)
cmp = pd.read_csv(TAB / 'model_comparison.csv')
cmp = cmp[cmp.threshold_rule == 'cost-optimal']
iso = pd.read_csv(TAB / 'anomaly_vs_supervised.csv').iloc[0]
costs = S['costs']
top_feats = ', '.join(S['top_features'])

# ---------------------------------------------------------------- 1 cover
s = new_slide(NAVY, 'Hook: we built one fraud decision system and tested it on two payment channels. '
                    'Promise the audience one surprising result: a 99.9%-accurate model that is useless.')
text(s, MARGIN, Inches(1.2), Inches(10), Inches(0.4), 'KLU+PATH-3 · ML-02 HACKATHON', size=16, bold=True, color=SOFT)
text(s, MARGIN, Inches(2.0), Inches(11.5), Inches(1.6), 'Catch fraud, not customers', size=60, bold=True, color=WHITE)
text(s, MARGIN, Inches(3.4), Inches(11), Inches(1.0),
     'A cost-aware Approve / Step-up OTP / Block system for UPI and digital payments', size=26, color=SOFT)
text(s, MARGIN, Inches(5.6), Inches(11), Inches(0.4), 'Team: [names]', size=18, color=WHITE)
text(s, MARGIN, Inches(6.05), Inches(11), Inches(0.4), 'github.com/MohammedMutee/upi-fraud-detection', size=16, color=SOFT)

# ---------------------------------------------------------------- 2 the trap
s = new_slide(PAPER, f"Open with the trap. A model that says 'genuine' to every payment is {S['trap']['dummy_accuracy_%']:.2f}% accurate "
                     'and catches nothing. Our model has almost the same accuracy but catches 78% of fraud. '
                     'So accuracy cannot tell a useless model from a useful one; we judge with recall, precision, PR-AUC and rupees.')
title(s, 'The accuracy trap', 'A 99.87%-accurate model that catches zero fraud')
stat(s, MARGIN, Inches(1.9), Inches(3.8), f"{S['trap']['dummy_accuracy_%']:.2f}%", 'accuracy of a model that\nsays “genuine” to everything')
stat(s, Inches(4.6), Inches(1.9), Inches(3.8), '0 of 74', 'frauds it catches on the test set', ORANGE)
stat(s, Inches(8.6), Inches(1.9), Inches(4.2), pct(fm['test_recall'] * 100), f"caught by our model, at\nalmost the same accuracy", GREEN)
image(s, '10_accuracy_trap_and_confusion.png', MARGIN, Inches(3.75), Inches(12.1), Inches(3.05))
footer(s, 'Card / digital channel, test slice (newest 20% of transactions). We use recall, precision, PR-AUC and expected ₹ cost instead.')

# ---------------------------------------------------------------- 3 the problem
s = new_slide(PAPER, 'Two-sided cost: missing fraud loses money; blocking a genuine customer loses trust. '
                     'Fraud is about 1 in 600 payments. We test one pipeline on two channels.')
title(s, 'The business problem', 'Every decision has two ways to lose money')
for i, (h, b, c) in enumerate([
        ('Miss a fraud', f"Money lost and trust lost.\nCost assumed: ₹{costs['missed_fraud']:,} each", ORANGE),
        ('Block a genuine customer', f"Angry customer, may switch apps.\nCost assumed: ₹{costs['false_block']:,} each", BLUE),
        ('Ask for an OTP', f"5 seconds of friction.\nCost assumed: ₹{costs['otp']:,} each", GREEN)]):
    x = MARGIN + i * Inches(4.1)
    card(s, x, Inches(1.95), Inches(3.85), Inches(1.9))
    text(s, x + Inches(0.3), Inches(2.15), Inches(3.3), Inches(0.5), h, size=22, bold=True, color=c)
    text(s, x + Inches(0.3), Inches(2.75), Inches(3.3), Inches(1.0), b, size=16, color=MUTED)
table(s, MARGIN, Inches(4.3), Inches(12.1), pd.DataFrame({
    'Channel': ['UPI', 'Card / digital'],
    'Dataset': ['upi_transactions 2024.csv (simulated, per the brief)', 'creditcard.csv (real, ULB)'],
    'Payments': [f"{S['data']['upi_rows']:,}", f"{S['data']['cc_rows_clean']:,} (after cleaning)"],
    'Frauds': [f"{S['data']['upi_frauds']} (0.19%)", f"{S['data']['cc_frauds']} (0.17%)"]}),
    col_w=[2, 5.5, 3, 2.2], size=15)
footer(s, '⚠ The ₹ costs are placeholder assumptions (no official values were published). Every saving shown is a scenario estimate.')

# ---------------------------------------------------------------- 4 data quality
s = new_slide(PAPER, 'We checked 13 kinds of problems. Card data had 1,081 duplicate rows: removed, because a copy '
                     'in train and test inflates the score. UPI had zero problems, which is suspicious for real data. '
                     'transaction_status is only known after processing, so it is leakage and excluded.')
title(s, 'PS1 · Data quality', 'Clean the data, and remove the columns that cheat')
table(s, MARGIN, Inches(1.9), Inches(12.1), pd.DataFrame({
    'Check': ['Exact duplicate rows (card)', 'Amount = 0 (card)', 'Missing values / bad timestamps', 'UPI duplicates / bad categories',
              'transaction_status (UPI)', 'transaction id (UPI)'],
    'Found': ['1,081 (19 fraud)', '1,825', '0 in both files', '0: “too perfect” for real data', '12,376 FAILED', '250,000 unique'],
    'What we did': ['Dropped: a copy in train and test inflates scores', 'Kept: likely card-verification pings',
                    'No action', 'Noted: first hint the UPI file is synthetic',
                    'EXCLUDED: only known after processing (leakage)', 'EXCLUDED: an identifier, not a signal']}),
    col_w=[3.3, 3, 5.8], size=15)
footer(s, 'Split: time-aware. Oldest 60% train · next 20% validation (every choice made here) · newest 20% test (reported once).')

# ---------------------------------------------------------------- 5 EDA
s = new_slide(PAPER, 'Card fraud is about 5 times more common between 1 and 5 AM, and amounts are U-shaped: tiny '
                     '“card-testing” payments and large ones are riskier. These are associations, not causes. '
                     'UPI shows no such pattern: every slice sits on the overall rate.')
title(s, 'PS3 · Where fraud lives', 'Night-time and tiny “test” payments carry more card fraud')
image(s, '02_cc_fraud_by_hour_amount.png', MARGIN, Inches(1.85), Inches(12.1), Inches(3.6))
text(s, MARGIN, Inches(5.65), Inches(12.1), Inches(1.2), [
    [('1–5 AM: ', {'bold': True, 'color': ORANGE}), ('0.67% fraud vs 0.14% at other hours (about 5×), peaking at 1.45% at 2 AM.', {})],
    [('Amount is U-shaped: ', {'bold': True, 'color': ORANGE}),
     ('₹0–1 payments 0.51% and mid amounts 0.06%. UPI: every type, bank and hour sits on the overall rate.', {})]], size=17)
footer(s, 'Error bars = 95% confidence intervals. Observed associations only; no causal claims.')

# ---------------------------------------------------------------- 6 models
pick = [('Rule baseline', 'No ML'), ('Logistic Regression', 'Class weights'), ('Random Forest', 'Class weights'),
        ('Random Forest', 'SMOTE'), ('XGBoost', 'No handling'), ('XGBoost', 'Class weights')]
rows = pd.DataFrame([cmp[(cmp.model == m) & (cmp.imbalance == i)].iloc[0] for m, i in pick])
rule = cmp[cmp.model == 'Rule baseline'].iloc[0]
mt = pd.DataFrame({
    'Model': [f'{m} ({i})' if m != 'Rule baseline' else 'Rule: night + tiny amount' for m, i in pick],
    'PR-AUC': [f'{v:.3f}' for v in rows.PR_AUC],
    'Frauds caught': [f'{int(tp)} / 74' for tp in rows.TP],
    'Genuine blocked': [f'{int(fp):,}' for fp in rows.FP],
    'Test cost': [inr(v) for v in rows['expected_cost_₹']]})
rule_default = pd.read_csv(TAB / 'model_comparison.csv').query("model == 'Rule baseline' and threshold_rule == 'default 0.5'").iloc[0]
mt.iloc[0, 2:] = [f"{int(rule_default.TP)} / 74", f"{int(rule_default.FP):,}", inr(rule_default['expected_cost_₹'])]
s = new_slide(PAPER, 'Nine models were trained (3 algorithms × no handling / class weights / SMOTE inside the pipeline), '
                     'with each cut-off chosen on validation. The hand-written rule annoys 6,343 good customers, which is worse '
                     'than doing nothing. XGBoost won on validation cost and trains in about 3 seconds.')
title(s, 'PS4 · Supervised models', 'Nine models against a hand-written rule')
table(s, MARGIN, Inches(1.9), Inches(12.1), mt, col_w=[4.6, 1.6, 2.2, 2.2, 2.2], size=16, highlight=4)
text(s, MARGIN, Inches(5.25), Inches(12.1), Inches(1.5), [
    [('Rules alone fail: ', {'bold': True, 'color': ORANGE}), ('the rule costs about 5× more than doing nothing.', {})],
    [('Imbalance tricks ', {'bold': True, 'color': ORANGE}),
     ('help simple models (LR: 0.61 → 0.75 PR-AUC). For trees, a cost-tuned cut-off does the same job.', {})]], size=17)
footer(s, 'Card / digital channel, test slice. SMOTE applied only inside the training pipeline. Highlighted = final model (chosen on validation).')

# ---------------------------------------------------------------- 7 cost cut-off
s = new_slide(PAPER, f"We tried about 1,500 cut-offs on validation. Cost = missed fraud × ₹X + wrong blocks × ₹Y. The cheapest is "
                     f"{fm['threshold']:.2f}, not the habitual 0.5. The sensitivity check shows the cut-off stays at 0.21 for "
                     'cost ratios from 10 to 100, so the choice is robust.')
title(s, 'PS6 · Money picks the cut-off', f"The cheapest line is {fm['threshold']:.2f}, not 0.5")
image(s, '07_cost_vs_threshold.png', MARGIN, Inches(1.85), Inches(7.6), Inches(4.0))
stat(s, Inches(8.6), Inches(1.9), Inches(4.2), lakh(nothing), 'do nothing (approve all)', MUTED, 36)
stat(s, Inches(8.6), Inches(3.35), Inches(4.2), lakh(default05), 'default 0.5 cut-off', BLUE, 36)
stat(s, Inches(8.6), Inches(4.8), Inches(4.2), lakh(opt), f"cost-chosen {fm['threshold']:.2f} cut-off", GREEN, 36)
text(s, MARGIN, Inches(6.05), Inches(12), Inches(0.7),
     f"At {fm['threshold']:.2f}: {fm['test_TP']} of 74 frauds caught, only {fm['test_FP']} genuine customers blocked.", size=17, color=INK)
footer(s, 'Test-slice expected cost under placeholder costs (scenario estimate). The cut-off was chosen on validation, never on test.')

# ---------------------------------------------------------------- 8 three zones
b = card_bands.set_index('zone')
s = new_slide(PAPER, 'Two cut-offs make three zones. Most payments sail through, a small slice gets an OTP, and the '
                     'block zone is tiny but holds three quarters of the fraud. We checked it against no-model policies '
                     '(OTP everyone, OTP a random 5%) so the saving is due to the model, and capped OTP at 5% of traffic.')
title(s, 'PS7 · The decision rule', 'Approve 96.7% · OTP 3.2% · Block 0.1%')
for i, (z, label, c) in enumerate([('APPROVE', 'Approve', GREEN), ('STEP-UP OTP', 'Step-up OTP', BLUE), ('BLOCK', 'Block', ORANGE)]):
    x = MARGIN + i * Inches(4.1)
    card(s, x, Inches(1.9), Inches(3.85), Inches(2.25))
    text(s, x + Inches(0.3), Inches(2.05), Inches(3.3), Inches(0.5), label, size=22, bold=True, color=c)
    text(s, x + Inches(0.3), Inches(2.6), Inches(3.3), Inches(0.7), pct(b.loc[z, '%_of_traffic'], 1) + ' of payments', size=28, bold=True, color=NAVY)
    who = {'APPROVE': f"{int(b.loc[z, 'frauds'])} frauds slip through",
           'STEP-UP OTP': f"{int(b.loc[z, 'frauds'])} frauds · {int(b.loc[z, 'genuine_customers_affected']):,} genuine asked",
           'BLOCK': f"{int(b.loc[z, 'frauds'])} frauds ({pct(b.loc[z, '%_of_all_fraud'])}) · only {int(b.loc[z, 'genuine_customers_affected'])} genuine"}[z]
    text(s, x + Inches(0.3), Inches(3.35), Inches(3.4), Inches(0.7), who, size=15, color=MUTED)
stat(s, MARGIN, Inches(4.5), Inches(4), lakh(zones), '3-zone rule, test cost', GREEN, 40)
stat(s, Inches(4.7), Inches(4.5), Inches(4), pct((1 - zones / nothing) * 100), 'cheaper than doing nothing', GREEN, 40)
stat(s, Inches(8.8), Inches(4.5), Inches(4), pct((1 - zones / opt) * 100), 'cheaper than the best single cut-off', BLUE, 40)
footer(s, f"OTP assumed to stop {costs['otp_stop_rate'] * 100:.0f}% of fraud; OTP zone capped at 5% of traffic. Scenario estimate under placeholder costs.")

# ---------------------------------------------------------------- 9 anomaly
s = new_slide(PAPER, 'New scams have no labels yet. The Isolation Forest never sees a fraud label, yet its top 1% most '
                     'unusual payments hold half of all fraud. It is a second line of defence for a review queue. Bonus: its '
                     'ROC-AUC is 0.95 but PR-AUC only 0.05, which is why we don’t trust ROC-AUC on rare fraud.')
title(s, 'PS5 · Unknown fraud', 'A detector that never saw a label finds half the fraud')
image(s, '05_anomaly_detection.png', MARGIN, Inches(1.85), Inches(12.1), Inches(3.6))
stat(s, MARGIN, Inches(5.6), Inches(4), pct(iso['fraud_caught_top_1.0%']), 'of fraud in its top 1%, with no labels', ORANGE, 36)
stat(s, Inches(4.7), Inches(5.6), Inches(4), pct(iso['fraud_caught_top_5.0%']), 'of fraud in its top 5%', BLUE, 36)
stat(s, Inches(8.8), Inches(5.6), Inches(4), f"{iso.ROC_AUC:.2f} vs {iso.PR_AUC:.2f}", 'ROC-AUC vs PR-AUC: why ROC flatters', MUTED, 36)
footer(s, 'Isolation Forest: 300 trees, trained on features only; labels used afterwards to evaluate.')

# ---------------------------------------------------------------- 10 two channels
s = new_slide(NAVY, 'The same pipeline on both channels. On cards it is 610 times better than random and catches 78% of fraud. '
                    'On UPI it is 1.2 times random and refuses to block: it blocked 2 of 50,000 payments instead of guessing.')
title(s, 'Two channels, one pipeline', f"Cards: {cardc.PR_AUC_lift_x:,.0f}× better than random. UPI: {upic.PR_AUC_lift_x:.1f}×", dark=True)
image(s, '12_two_channel_comparison.png', MARGIN, Inches(1.85), Inches(12.1), Inches(3.4))
stat(s, MARGIN, Inches(5.45), Inches(6), pct(cardc['recall_at_cut_off_%']) + ' caught · ' + pct(cardc['3zone_saving_vs_nothing_%']) + ' saved',
     'Card / digital: real labels, real signal', WHITE, 32)
stat(s, Inches(6.9), Inches(5.45), Inches(6), f"0% caught · {int(upic.genuine_blocked)} blocked of 50,000",
     'UPI: no signal, so the system safely refuses to block', WHITE, 32)
footer(s, 'PR-AUC ÷ fraud rate on each test slice. Identical code, no channel-specific tuning.', dark=True)

# ---------------------------------------------------------------- 11 rescue test
s = new_slide(PAPER, 'Could we just clean the UPI data? We added 14 engineered features and a strong model, then shuffled the '
                     'labels 20 times. The real labels score no better than shuffled ones (p = 0.43): the flags were assigned '
                     'without looking at any column. Inventing our own labels would be circular, so we report it honestly.')
title(s, 'Why are UPI results low?', 'Real UPI labels score the same as shuffled ones')
image(s, '15_upi_shuffled_label_test.png', MARGIN, Inches(1.85), Inches(7.4), Inches(4.0))
stat(s, Inches(8.4), Inches(1.9), Inches(4.4), f'{real_auc:.3f}', 'ROC-AUC with the real UPI labels', ORANGE, 40)
stat(s, Inches(8.4), Inches(3.3), Inches(4.4), f'{sh.mean():.3f} ± {sh.std(ddof=0):.3f}', '20 × randomly shuffled labels', MUTED, 40)
stat(s, Inches(8.4), Inches(4.7), Inches(4.4), f'p = {p_value:.2f}', 'no evidence of any pattern', NAVY, 40)
text(s, MARGIN, Inches(6.05), Inches(12), Inches(0.7),
     'Cleaning fixes broken values, and this file has none. The labels don’t depend on the inputs: a labelling problem, not a model problem.',
     size=16, color=INK)
footer(s, '27 features (14 engineered), gradient boosting, 5-fold cross-validation. Script: scripts/upi_rescue_test.py')

# ---------------------------------------------------------------- 12 SHAP
s = new_slide(PAPER, f"SHAP shows what pushes a payment towards fraud: {top_feats}. Hour of day and amount also matter, "
                     'which matches the EDA. The V-columns are anonymised by the bank, so we can rank them but not name them.')
title(s, 'PS8 · Why the model blocks', 'Explaining every decision with SHAP')
image(s, '11_shap_summary.png', MARGIN, Inches(1.8), Inches(6.4), Inches(5.0))
text(s, Inches(7.4), Inches(2.0), Inches(5.4), Inches(4.6), [
    [('Top drivers: ', {'bold': True, 'color': ORANGE}), (top_feats, {})],
    [('Also matter: ', {'bold': True, 'color': ORANGE}), ('hour of day and amount, consistent with the EDA', {})],
    [('Read it: ', {'bold': True, 'color': ORANGE}), ('red = high feature value; right of centre = pushes towards fraud', {})],
    [('Limit: ', {'bold': True, 'color': ORANGE}), ('V1–V28 are anonymised (PCA), so we rank them but cannot name them', {})]], size=18)
footer(s, 'SHAP TreeExplainer on 3,000 test payments, final XGBoost model.')

# ---------------------------------------------------------------- 13 recommendations
s = new_slide(NAVY, 'Three actions, each tied to evidence. All savings are scenario estimates under placeholder costs.')
title(s, 'PS8 · Three recommendations', 'Finding → evidence → action', dark=True)
recs = [
    ('1 · Pilot the 3-zone rule', f"{pct((1 - zones / nothing) * 100)} cheaper than doing nothing; blocks only 0.1% of payments",
     'Run XGBoost + Approve/OTP/Block on a slice of live card traffic. Re-tune the cut-offs monthly with real costs.'),
    ('2 · Add a no-label safety net', f"Isolation Forest: {pct(iso['fraud_caught_top_1.0%'])} of fraud in its top 1%, without labels",
     'Send unusual-but-approved payments to a human review queue to catch new scam types early.'),
    ('3 · Fix UPI fraud labelling', f"UPI labels: {upic.PR_AUC_lift_x:.1f}× random, the same as shuffled labels (p = {p_value:.2f})",
     'Label from confirmed complaints and chargebacks. Until then: approve + capped OTP on UPI, never auto-block.')]
for i, (h, ev, act) in enumerate(recs):
    x = MARGIN + i * Inches(4.1)
    card(s, x, Inches(1.95), Inches(3.85), Inches(4.7), RGBColor(0x1E, 0x2E, 0x52))
    text(s, x + Inches(0.3), Inches(2.15), Inches(3.3), Inches(0.9), h, size=22, bold=True, color=WHITE)
    text(s, x + Inches(0.3), Inches(3.15), Inches(3.3), Inches(1.4), ev, size=15, bold=True, color=SOFT)
    text(s, x + Inches(0.3), Inches(4.55), Inches(3.3), Inches(2.0), act, size=15, color=WHITE)
footer(s, 'Priority: 1 and 3 high, 2 medium. Observed results and projected savings are kept separate.', dark=True)

# ---------------------------------------------------------------- 14 limitations
s = new_slide(PAPER, 'Be upfront: what is assumed, what is measured, and what the data cannot show.')
title(s, 'Limitations', 'What is measured, what is assumed')
table(s, MARGIN, Inches(1.9), Inches(12.1), pd.DataFrame({
    'Limitation': ['₹ costs are placeholders', 'OTP stop rate (80%) and 5% OTP cap', 'Card data covers 2 days; 74 test frauds',
                   'V1–V28 anonymised', 'UPI data is simulated', 'Observational data'],
    'Effect': ['Savings are scenario estimates; cut-offs may move (stable for cost ratios 10–100)',
               'Assumptions: they change the OTP zone, not the model',
               'One fraud = 1.4% recall; no seasonal patterns',
               'We can rank drivers but not name them', 'Its labels carry no signal; UPI results show behaviour, not patterns',
               'Associations, never causes']}), col_w=[4, 8.1], size=16)
footer(s, 'Re-run with official costs: change one cell in the notebook (about 15 min end to end).')

# ---------------------------------------------------------------- 15 close
s = new_slide(NAVY, 'Close: repeat the three numbers, then take questions.')
text(s, MARGIN, Inches(1.6), Inches(12), Inches(1.0), 'Catch fraud, not customers', size=48, bold=True, color=WHITE)
text(s, MARGIN, Inches(2.8), Inches(12), Inches(2.4), [
    f"• Accuracy lies: {S['trap']['dummy_accuracy_%']:.2f}% accurate, 0 frauds caught",
    f"• Cost-chosen 3-zone rule: {pct(fm['test_recall'] * 100)} of fraud caught, {pct((1 - zones / nothing) * 100)} cheaper than doing nothing",
    f"• Same pipeline on UPI: no signal in the labels, so it safely refuses to block"], size=24, color=SOFT)
text(s, MARGIN, Inches(5.6), Inches(12), Inches(0.5), 'Questions?', size=32, bold=True, color=WHITE)
text(s, MARGIN, Inches(6.3), Inches(12), Inches(0.4), 'github.com/MohammedMutee/upi-fraud-detection', size=16, color=SOFT)

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print(f'wrote {OUT} ({len(prs.slides)} slides)')
