# 🏦 Real-Time UPI / Digital Payment Fraud Detection

KLU+Path-3 · ML-02 Hackathon.

**Team: Foundry AI · Group 3** ([TEAM.md](TEAM.md)): Mohammed Mutee Ruknuddin · Buradakavi Durga Sri Varshini · P Annapurna · Kondaveti Tejaswanth · D Siva Dhanesh

This system scores each payment and decides whether to **approve**, ask for a **step-up OTP**, or **block** it. The cut-offs are chosen by rupee cost, not by habit.

## Two channels, one pipeline

| | Card / digital (`creditcard.csv`) | UPI (`upi_transactions 2024.csv`) |
|---|---|---|
| Final model | XGBoost | Random Forest (SMOTE) |
| PR-AUC vs random | **0.796 = 610× random** | 0.0023 = 1.2× random |
| ROC-AUC | 0.982 | 0.526 |
| Fraud caught at the cost cut-off | **78%** | 0% (the cost rule refuses to block without a signal) |
| 3-zone saving vs doing nothing | **81%** | 1.9% (noise) |

The identical pipeline learns real fraud patterns on the card channel and correctly finds none on the UPI channel. The brief describes the UPI data as *simulated*, and its labels carry no signal (see notebook Sections 3b, 9 and 10).

## Card / digital channel in detail (test slice, seed 42, placeholder costs)

| | Value |
|---|---|
| Fraud rate | 0.167% (473 of 283,726 card transactions; about 1 in 600) |
| Do-nothing model | **99.87% accuracy, 0% of fraud caught** (the accuracy trap) |
| Final model | **XGBoost**, cut-off 0.21 chosen by cost on validation |
| Test PR-AUC / ROC-AUC | 0.796 / 0.982 |
| Fraud caught / genuine blocked | 58 of 74 (78%) / 19 of 56,672 |
| Anomaly detector (no labels) | Isolation Forest finds 50% of fraud in its top 1% |
| 3-zone rule | Approve 96.6% · OTP 3.2% · Block 0.11% of traffic |
| Expected cost on test | **₹1.39 lakh** (3 zones) vs ₹1.85 lakh (0.5 cut-off) vs ₹7.40 lakh (do nothing) |

> ⚠️ The ₹ costs (₹10,000 per missed fraud, ₹500 per false block, ₹10 per OTP) and the 80% OTP stop rate are **placeholders**. Replace them with the organiser values in the first code cell of the notebook and re-run. Every savings figure is a **scenario estimate**.

## Datasets

| File | Channel | Rows | Fraud | Notes |
|---|---|---|---|---|
| `upi_transactions 2024.csv` | UPI | 250,000 | 480 (0.192%) | Readable fields (type, merchant, bank, device, ₹ amount). Simulated, and **the labels look random** |
| `creditcard.csv` (ULB) | Card / digital | 284,807 | 492 (0.173%) | Real data with a strong signal; the features are anonymised (PCA) |

The full reasoning for each step is in the notebook's 📌 *What / Why / How / Impact* boxes.

## Setup

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

**Data (not in Git):** the two CSV files must be in the project root, next to this README. They're excluded from the repo because `creditcard.csv` is 144 MB, which is over GitHub's 100 MB file limit.
- `creditcard.csv`: the ULB Credit Card Fraud dataset (Kaggle: *mlg-ulb/creditcardfraud*), or the copy in the hackathon handout
- `upi_transactions 2024.csv`: from the hackathon handout

## How to run

```bash
# 1. The full analysis (about 15 minutes on 8 cores). It writes every figure, table and the saved model
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace fraud_analysis.ipynb
#    ...or open it interactively:  ../.venv/bin/jupyter notebook fraud_analysis.ipynb

# 2. The optional live demo (run the notebook first; it needs models/final_model.joblib)
cd ..
.venv/bin/streamlit run app/streamlit_demo.py
```

To change the notebook text or code, edit `scripts/build_notebook.py` and regenerate it:

```bash
.venv/bin/python scripts/build_notebook.py notebooks/fraud_analysis.ipynb   # then execute it (step 1)
.venv/bin/python scripts/refresh_text.py notebooks/fraud_analysis.ipynb     # text-only edits: keeps outputs, no re-run
```

## Submission deliverables

| Deliverable | File |
|---|---|
| Notebook (all 8 problem statements, both channels) | [`notebooks/fraud_analysis.ipynb`](notebooks/fraud_analysis.ipynb) |
| Business report | [`report/business_report.md`](report/business_report.md) · [PDF](report/business_report.pdf) |
| Presentation (15 slides, speaker notes included) | [`presentation/fraud_detection_deck.pptx`](presentation/fraud_detection_deck.pptx) · [PDF](presentation/fraud_detection_deck.pdf) |
| Q&A preparation | [`presentation/qa_prep.md`](presentation/qa_prep.md) |
| Speaker script (who says what, slide by slide) | [`presentation/speaker_script.md`](presentation/speaker_script.md) |
| Model comparison table | [`outputs/tables/model_comparison.csv`](outputs/tables/model_comparison.csv) · [UPI](outputs/tables/upi_model_comparison.csv) |
| Cost analysis | [`outputs/figures/07_cost_vs_threshold.png`](outputs/figures/07_cost_vs_threshold.png) · [`cost_sensitivity.csv`](outputs/tables/cost_sensitivity.csv) |
| Risk-score rule | [`outputs/tables/risk_bands_test.csv`](outputs/tables/risk_bands_test.csv) · [`policy_costs.csv`](outputs/tables/policy_costs.csv) |
| Live demo | `app/streamlit_demo.py` (card and UPI channels) |

To rebuild the slides and report after a re-run (they read their numbers from `outputs/`):

```bash
.venv/bin/python scripts/build_slides.py
soffice --headless --convert-to pdf --outdir presentation presentation/fraud_detection_deck.pptx
.venv/bin/python scripts/build_report_pdf.py
```

## Project layout

```
├── creditcard.csv, upi_transactions 2024.csv   input data
├── notebooks/fraud_analysis.ipynb              the main analysis (PS1–PS8 on both channels), seed 42
├── scripts/build_notebook.py                   source that generates the notebook
├── scripts/upi_rescue_test.py                  UPI shuffled-label test (also run inside the notebook)
├── scripts/build_slides.py, build_report_pdf.py   build the deck and the report PDF from outputs/
├── report/                                     business report (.md + .pdf)
├── presentation/                               slide deck (.pptx + .pdf) and Q&A prep
├── app/streamlit_demo.py                       Approve / OTP / Block demo (card + UPI)
├── models/final_model.joblib, upi_model.joblib final models + cut-offs + Isolation Forests
├── outputs/figures/                            15 charts (01_class_imbalance … 15_upi_shuffled_label_test)
├── outputs/tables/                             data quality, model comparison, bands, costs …
└── outputs/summary.json                        key numbers in one place
```

## Method (one line per problem statement)

1. **Data quality:** dropped 1,081 duplicate rows, and excluded `transaction id` and `transaction_status` (leakage).
2. **Accuracy trap:** time-aware split (60/20/20 for train/validation/test), and a do-nothing baseline.
3. **EDA:** fraud rate with 95% confidence intervals by hour, amount and type. Chi-square tests with a Bonferroni correction on the UPI data.
4. **Supervised:** LR, RF and XGBoost × (no handling, class weights, SMOTE inside an `imblearn` Pipeline), plus a rule baseline.
5. **Anomaly:** Isolation Forest fitted without labels and evaluated afterwards.
6. **Cost cut-off:** a cost sweep on validation, then a sensitivity check over 15 ₹X/₹Y scenarios.
7. **Risk rule:** a grid search over (T1, T2) on validation, reported on test.
8. **Explain:** SHAP, then 3 findings turned into 3 recommendations, plus limitations.
9. **UPI channel:** steps 4–8 repeated with identical code.
10. **Two-channel comparison:** PR curves and "× better than random".

The 3-zone rule is checked against two "no model" policies (*OTP everyone*, *OTP a random 5%*), and the OTP zone is capped at 5% of traffic.

## Rules followed

- No resampling before the split; SMOTE only runs inside the training pipeline.
- No leakage columns.
- Cut-offs and model choice use validation only; test is used once.
- No causal claims; patterns are reported as associations.
- Projected savings are labelled as scenario estimates.
