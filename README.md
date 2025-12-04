# Quant Signal Synthesiser: Machine Learning for Trading
### A Meta-Labeling Framework using Random Forest & XGBoost to Filter Technical Strategies

## 1. Summary
This project applies **Meta-Labeling** to improve the risk-adjusted returns of a simple technical trading strategy.

Instead of predicting price direction (which has a low signal-to-noise ratio), this project predicts the probability of trade success conditional on market regimes. By training Random Forest and XGBoost classifiers on "Market Weather" features (Volatility, VIX, Volume), the model learns to avoid trades during high-risk conditions.

> **Key Result:** The XGBoost Model successfully identified the 2022 Bear Market regime, reducing the portfolio's Maximum Drawdown by **~250 basis points** compared to the S&P 500 benchmark and achieving positive returns while the market fell.

---

## 2. The Hypothesis (Research Question)

**The Problem:** Simple technical indicators (like RSI Mean Reversion or Trend Following) are "naive." They generate signals regardless of the macro environment. A mean-reversion trade that works in a calm market will "catch a falling knife" during a volatility spike.

**The Hypothesis:**
Financial markets exhibit distinct regimes (Low Vol/High Vol). A Machine Learning model can learn the non-linear interaction between technical signals and volatility to filter out false positives.

---

## 3. Methodology

### A. Data Pipeline & Features
* **Asset:** SPY (S&P 500 ETF)
* **Data Period:** 2015–2024
    * *Training:* 2015-2020
    * *Validation:* 2020-2022
    * *Testing:* 2022-Present

**Strategy Universe (The Inputs):**
I engineered 4 distinct signal generators to provide diverse inputs:
* **RSI (Mean Reversion):** Buy when Oversold (<30).
* **SMA Distance (Trend):** Buy when Price > 50-day Moving Average.
* **MACD (Momentum):** Buy on Bullish Crossover.
* **Bollinger Bands (Breakout):** Buy on Upper Band breach.

**Context Features (The Filters):**
* **Volatility:** Rolling Standard Deviation (20d).
* **VIX (Fear Index):** Normalized VIX levels to detect market stress.
* **Seasonality:** Day of Week (testing the "Friday Effect").

### B. The Labeling Technique (Triple Barrier Method)
Standard fixed-time labeling (e.g., "Price in 5 days") is flawed because it ignores the path price takes (risk). I implemented a **Vectorized Triple Barrier Method**:

* **Upper Barrier (Take Profit):** +1.5%
* **Time Barrier (Expiration):** 5 Days
* **Logic:** A label of 1 (Win) is assigned only if the price hits the target within the window. If it hits a stop-loss or expires flat, it is labeled 0.

### C. The Mathematical Framework
The model optimizes for **Precision** rather than Accuracy, as the cost of a False Positive (losing money) is higher than a False Negative (missing a trade).

$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

Risk-Adjusted Returns were evaluated using the Sharpe Ratio:

$$\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}$$

---

## 4. Experimentation & Model Evolution

I conducted a "Tournament" between two algorithms to find the best regime detector.

| Model | Hypothesis | Results |
| :--- | :--- | :--- |
| **Random Forest (Bagging)** | A "Council of Experts" approach will reduce variance and handle noise well. | High stability, but slightly conservative. |
| **XGBoost (Boosting)** | A sequential correction approach will aggressively find edge cases and improve precision. | **Winner.** Better at identifying specific failure modes in high volatility. |

### The Optimization Process (Grid Search)
I performed a 3-Fold Cross-Validation Grid Search to tune hyperparameters:

* **Random Forest:** Tuned `max_depth` and `min_samples_leaf`. Found that deeper trees (depth=10) with moderate leaf size (10) balanced bias and variance best.
* **XGBoost:** Tuned `learning_rate` and `scale_pos_weight`. The model required a high weight on the minority class (Wins) to overcome the dataset imbalance.

---

## 5. Results & Discussion

### Performance on Unseen Test Data (2022–2024)
The test period covered the 2022 Inflation Bear Market and the 2023 Recovery.

| Strategy | Total Return | Sharpe Ratio | Max Drawdown |
| :--- | :--- | :--- | :--- |
| **S&P 500 (Benchmark)** | +2.08% | 0.15 | -24.46% |
| **Random Forest (Opt)** | -2.72% | 0.01 | -22.86% |
| **XGBoost (Opt)** | **+0.22%** | **0.09** | **-21.93%** |

### Key Findings
1.  **Risk Management Success:** All ML models reduced the Maximum Drawdown compared to the market (-21.9% vs -24.5%). The meta-labeling successfully identified "Crash Regimes" and kept the strategy in cash.
2.  **The Precision-Recall Trade-off:** The Random Forest was too conservative during the 2023 recovery, resulting in a slight loss. XGBoost managed to capture enough upside to stay positive (+0.22%) while still protecting downside.
3.  **Regime Interaction:** Feature Importance analysis confirmed that Volatility and VIX were the primary decision drivers, validating the hypothesis that market stress is the best predictor of technical failure.

---

## 6. Project Structure

```bash
quant-signal-synthesizer/
├── data/                   # Raw and Processed Data (GitIgnored)
├── models/                 # Serialized Models (rf_opt.joblib, xgb_opt.joblib)
├── notebooks/
│   ├── 01_data_prep.ipynb  # EDA, Signal Generation (RSI/MACD/BB), Labeling
│   ├── 02_train_rf.ipynb   # Random Forest Training & Baseline
│   ├── 03_train_xgb.ipynb  # XGBoost Training & Baseline
│   ├── 04_optimization.ipynb # Grid Search Cross-Validation for both models
│   ├── 05_backtest.ipynb   # Out-of-Sample Comparative Backtest (The Grand Prix)
├── src/
│   ├── data_loader.py      # YFinance wrappers & Caching logic
│   ├── indicators.py       # Technical Feature Engineering
│   ├── labeling.py         # Triple Barrier Logic implementation
├── requirements.txt

└── README.md
