# Speaker Script: “Catch Fraud, Not Customers”

One story, five speakers. Each part tells **what** we did, **why**, **how**, and **what happened (the impact)**.
Read it a few times out loud. You don’t need to say it word for word; tell it like a story in your own words.

| Order | Speaker | Slides | About |
|---|---|---|---|
| 1 | **D Siva Dhanesh** | 1–2 | about 1.4 min |
| 2 | **Buradakavi Durga Sri Varshini** | 3–5 | about 2.2 min |
| 3 | **P Annapurna** | 6–7 | about 2.0 min |
| 4 | **Kondaveti Tejaswanth** | 8–10 | about 2.0 min |
| 5 | **Mohammed Mutee Ruknuddin** | 11–15 | about 2.2 min |

**Tips for everyone**
- Speak slowly. Pause after every big number and let it sink in.
- Point at the slide when you say a number, so people look at it.
- End your part with the handover line, so the story keeps flowing.
- If you forget a line, just say the big number on the slide and what it means.

---

## 🎤 D Siva Dhanesh

### Slide 1 · Catch fraud, not customers

Good morning, everyone. We are a team of five: Mohammed Mutee Ruknuddin, Buradakavi Durga Sri Varshini, P Annapurna, Kondaveti Tejaswanth, and me, D Siva Dhanesh. Our project is about stopping fraud in UPI and digital payments. Our title is “Catch fraud, not customers”. That means: stop the thieves, but don’t trouble honest people.

### Slide 2 · Every decision has two ways to lose money

Think about paying at a shop with your phone. In less than one second, the payment app must decide one of three things: let the payment go, ask you for an OTP, or block it. Both mistakes cost money. If the app misses a fraud, the money is gone; we assumed ₹10,000 for that. If it blocks an honest customer, that customer gets angry and may leave the app; we assumed ₹500. Asking for an OTP is only a small trouble; we assumed ₹10. And fraud is very rare: only about 1 in every 600 payments. We worked with two datasets: about 2.5 lakh UPI payments and about 2.8 lakh card payments. Now Varshini will show you why the usual way of scoring a model is a trap.

---

## 🎤 Buradakavi Durga Sri Varshini

### Slide 3 · A 99.87%-accurate model that catches zero fraud

Thank you, Siva. Here is the trap. We built a lazy model that says “genuine” to every single payment. It is 99.87% accurate. That sounds great! But it catches zero of the 74 frauds. Why? Because almost every payment is genuine, so saying “genuine” is almost always right. Our real model has nearly the same accuracy, but it catches 78% of the fraud. So accuracy can’t tell a useless model from a useful one. Instead we measure recall, which means how many frauds we caught; precision, which means how many alarms were real; and the money lost.

### Slide 4 · Clean the data, and remove the columns that cheat

Before teaching any model, we checked the data, because bad data gives bad answers. We found 1,081 duplicate payments and removed them. If the same payment is in both the practice data and the exam data, the model looks better than it really is. We also removed a cheat column called transaction status. It tells whether a payment succeeded, but you only know that after the payment, so using it would be cheating. We split the data by time: old payments to learn from, the newest as the final exam, just like real life. And the UPI file had zero errors: too perfect for real data, our first hint it was made by a computer.

### Slide 5 · Fraud peaks at night and on tiny “test” payments

Next we asked: where does fraud happen? In the card data, fraud is about 5 times more common between 1 and 5 in the morning. Very tiny payments are also risky, because thieves test a stolen card with a tiny payment first. In the UPI data we found no pattern at all. And we say “linked with”, never “causes”. Now Annapurna will explain how we built our models.

---

## 🎤 P Annapurna

### Slide 6 · Nine models against a hand-written rule

Thank you, Varshini. A model is a program that learns from past examples and gives every new payment a risk score from 0 to 1. First we built simple baselines to compare against: the lazy model you just saw, and a hand-written rule: flag night-time or tiny payments. That rule caught some fraud, but it wrongly flagged 6,343 honest customers, so it costs more than doing nothing. Then we compared three real models. Logistic Regression draws one straight line. Random Forest is 200 decision trees that vote. XGBoost builds trees one after another, each fixing the last one’s mistakes. Because fraud is rare, we trained each one three ways: normally, with class weights so fraud counts more, and with SMOTE, which makes extra practice fraud examples. That is nine models. XGBoost won: it lost the least money on our choosing data, and it trains in about 3 seconds.

### Slide 7 · The cheapest line is 0.21, not 0.5

Every model gives a score. But where do we draw the line between “let it go” and “stop it”? Most people use 0.5, just out of habit. We used money instead. We tried about 1,500 different lines and picked the one that loses the least money. The best line is 0.21, not 0.5. Doing nothing loses ₹7.40 lakh. The 0.5 line loses ₹1.85 lakh. Our line loses only ₹1.70 lakh. At this line we catch 58 of 74 frauds and block only 19 honest customers. Even if the costs change a lot, this line stays the best. Now Tejaswanth will show how we turn this score into real decisions.

---

## 🎤 Kondaveti Tejaswanth

### Slide 8 · Approve 96.6% · OTP 3.2% · Block 0.1%

Thank you, Annapurna. Only “allow” or “block” is too strict, so we used two lines to make three zones. Low score: approve. Middle score: ask for an OTP. High score: block. Here is the result. 96.6% of payments go straight through. 3.2% get an OTP. Only 0.1% are blocked, and that tiny block zone holds 76% of all the fraud, while only 9 honest customers are blocked. The total loss is ₹1.39 lakh. That is 81% less than doing nothing. We also made sure the saving comes from the model, not just from cheap OTPs: sending an OTP to everyone would cost ₹7.15 lakh.

### Slide 9 · A detector that never saw a label finds half the fraud

But what about new scams that nobody has seen yet? They have no labels. So we built an Isolation Forest. It is never told what fraud is. It just looks for weird payments: it keeps cutting the data randomly, and weird payments get separated very quickly. Even without any labels, its top 1% weirdest payments held 50% of all the fraud, and it trains in about 1 second. We use it as a second line of defence: weird payments go to a person to check.

### Slide 10 · Explaining every decision with SHAP

People will ask: why did the model block my payment? We used a tool called SHAP. It shows which details pushed the score up or down. The strongest ones are hidden bank columns called V4, V14, V12, V10, V11, and also the hour of the day and the amount. That matches what Varshini showed: night-time and tiny test payments. Now Mutee will share our most surprising discovery.

---

## 🎤 Mohammed Mutee Ruknuddin

### Slide 11 · Cards: 610× better than random. UPI: 1.2×

Thank you, Tejaswanth. We ran the exact same system, with the same code, on both channels. On card payments it is 610 times better than guessing and catches 78% of fraud. On UPI it is only 1.2 times better than guessing and catches nothing. Same code, very different results. So the difference must come from the data, not from our method.

### Slide 12 · Real UPI labels score the same as shuffled ones

Is the UPI data bad, or is our model weak? We tested it. We added 14 extra clue columns and a strong model, then shuffled the fraud labels randomly 20 times, so they had no pattern at all. The real labels scored 0.499. The shuffled labels scored 0.495. They are the same. This means the fraud flags in the UPI file were given out like a coin toss. Cleaning cannot fix that, and making our own labels would be cheating. The good news: our system noticed there was no signal, and it blocked only 2 out of 50,000 payments instead of guessing.

### Slide 13 · Finding → evidence → action

Our three recommendations. One: test the three-zone system on a small part of card payments; it loses 81% less money. Two: add the no-label detector as a safety net, so new scams go to human review. Three: fix how UPI fraud is labelled. Until then, UPI should never block automatically.

### Slide 14 · What is measured, what is assumed

Our limits, honestly: the rupee costs are our own assumptions, the card data covers only two days, and the UPI data is computer-made. Everything we found is a link, not proof of a cause.

### Slide 15 · Summary and questions

To sum up: accuracy can lie. Our system catches 78% of fraud and loses 81% less money, and on UPI it was smart enough not to guess. Everything is on GitHub. Thank you! We are happy to take your questions.

---

*Every number in this script is read from the analysis results (`outputs/`), so it always matches the slides. Rupee figures use our assumed costs and are scenario estimates.*
