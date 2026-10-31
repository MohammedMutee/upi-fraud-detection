# Q&A Prep: Questions the Judges Will Probably Ask

Short answers first, then the evidence. Every number comes from `outputs/summary.json` and `outputs/tables/`.

---

## About metrics

**Q: Your model is 99.9% accurate. Isn't that great?**
No, and that's our opening point. A model that approves *everything* is 99.87% accurate and catches **zero** fraud. Fraud is 1 in 600 payments, so accuracy mostly measures how easy the genuine payments are. We judge with recall, precision, PR-AUC and expected rupee cost.

**Q: Why PR-AUC instead of ROC-AUC?**
ROC-AUC looks good even when a model raises many false alarms, because there are so many genuine payments. Our own proof: the Isolation Forest has **ROC-AUC 0.95 but PR-AUC 0.05**. PR-AUC shows what the fraud team actually lives with.

**Q: Why not the default 0.5 cut-off?**
0.5 is a habit, not a decision. We priced each cut-off: missed fraud × ₹X + wrong blocks × ₹Y. The cheapest is **0.21**, which costs ₹1.70 lakh against ₹1.85 lakh at 0.5 on the test slice. For Logistic Regression with class weights, 0.5 is a disaster: it blocks **2,921** good customers.

---

## About the model

**Q: Why XGBoost when Random Forest has a higher test PR-AUC (0.819 vs 0.796)?**
We fixed the selection rule *before* looking at test: the lowest **validation** cost, ties broken by validation PR-AUC. XGBoost and Random Forest tied on validation cost, and XGBoost won the tie (0.786 vs 0.767). Switching now because of test numbers would be tuning on the test set, which the rules forbid. XGBoost is also about 14× faster to train.

**Q: You chose XGBoost with *no* imbalance handling. Isn't that wrong for imbalanced data?**
We tested it rather than assuming. Class weights and SMOTE helped Logistic Regression a lot (PR-AUC 0.61 → 0.75). For tree models, once the cut-off is tuned by cost, the cut-off already does the rebalancing. The evidence is in the comparison table.

**Q: Where exactly did you apply SMOTE?**
Only inside an `imblearn` Pipeline, so it runs during `fit` on **training rows only**. Validation and test rows are never resampled.

**Q: Isn't there leakage?**
- We excluded `transaction_status`, because SUCCESS/FAILED is only known *after* processing, and we excluded the transaction ID.
- We removed 1,081 duplicate card rows that could sit on both sides of the split.
- The split is by time: the model trains on the past and is tested on the future.

**Q: Why should we trust the explanation if V1–V28 are hidden?**
SHAP ranks what drives each decision. Hour of day and amount show up among the top drivers, which matches the EDA (night and tiny "card-testing" payments). We can rank the V-features but not name them. That's a limit of the dataset, and we say so.

---

## About UPI (the toughest topic)

**Q: Why are your UPI results so bad?**
Because the UPI labels carry no pattern. We proved it: with 14 extra engineered features and a strong model, the real labels score **ROC-AUC 0.499**, while labels we **shuffled at random** score **0.495 ± 0.013** (p = 0.43). The fraud flags were assigned without regard to any recorded column. The brief itself calls this dataset simulated.

**Q: Couldn't you clean the data or engineer better features?**
We tried exactly that (Section 9b of the notebook). Cleaning fixes broken *values*, and this file has none. Its problem is that the answer column doesn't depend on the input columns. No feature can recover a pattern that isn't there.

**Q: Why not create your own fraud labels with rules?**
Then the model would just learn our own rule back and score 99%. That's circular and meaningless, and the rules forbid changing the official outcome.

**Q: So is your system useless on UPI?**
No. It behaved correctly. Seeing no signal, the cost rule chose **not to block** (2 blocks in 50,000 payments) instead of troubling customers at random. Our recommendation: fix UPI labelling (use confirmed complaints and chargebacks), and until then run UPI on approve + a capped OTP, with no auto-blocking.

**Q: Then why use UPI at all?**
The challenge is "UPI / digital payment" fraud. Running the identical pipeline on both channels shows the method works (610× better than random on cards). The UPI gap comes from the data, not the model.

---

## About costs and the decision rule

**Q: Where do your rupee costs come from?**
No official values were published, so we used clearly labelled placeholders: ₹10,000 per missed fraud, ₹500 per wrong block, ₹10 per OTP. Every saving is a scenario estimate. The **sensitivity check** shows the cut-off stays at 0.21 for every ₹X/₹Y ratio from 10 to 100, so the decision is robust. To use real costs, change one notebook cell and re-run.

**Q: Isn't the saving just because OTPs are cheap?**
Good catch, and we tested it. Two "no model" policies: *OTP everyone* costs ₹7.15 lakh, and *OTP a random 5%* costs ₹7.60 lakh. Our 3-zone rule costs **₹1.39 lakh**. The saving comes from the model choosing the *right* payments.

**Q: How did you pick the three zones?**
We searched about 600 × 600 pairs of cut-offs (T1, T2) on validation and kept the cheapest. The OTP zone is capped at 5% of traffic, because no bank would challenge every customer. Result: approve 96.7%, OTP 3.2%, block 0.11%. The block zone holds 76% of all fraud and wrongly blocks only 9 genuine customers.

**Q: What's the 80% OTP stop rate based on?**
It's an assumption we can't measure from this data, and we label it as one. A different rate changes the OTP zone's cost, not the model.

---

## About anomaly detection

**Q: Why Isolation Forest and not DBSCAN?**
Isolation Forest gives a ranking score, which the brief's "top-ranked anomalies" question needs. DBSCAN only says outlier yes or no. Isolation Forest is also fast: it trained in 1 second.

**Q: How good is it without labels?**
50% of all test fraud lands in its top 1% of unusual payments, and 84% in its top 5%. The supervised model gets 88% in its top 1% with twice the precision, so we use the anomaly score as a **second line** that feeds a review queue for new scams.

---

## About reproducibility

**Q: Can we reproduce your results?**
Yes. A fixed seed (42), the same numbers on every run, one notebook generated from `scripts/build_notebook.py`, and a README with setup steps. The full run takes about 15 minutes.

**Q: What are your main limitations?**
- Placeholder costs and an assumed OTP stop rate.
- The card data covers only 2 days, with 74 test frauds, so one fraud equals 1.4 points of recall.
- The card features are anonymised.
- The UPI data is simulated.
- Everything is an association, not a cause.
