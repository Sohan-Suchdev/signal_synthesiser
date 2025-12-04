# Quant Signal Synthesiser: Machine Learning for Trading
### A Meta-Labeling Framework using Random Forest & XGBoost to Filter Technical Strategies

## 1. Summary
This project applies **Meta-Labeling** to improve the risk-adjusted returns of a simple technical trading strategy.

Instead of predicting price direction (which has a low signal-to-noise ratio), this project predicts the probability of trade success conditional on market regimes. By training Random Forest and XGBoost classifiers on "Market Weather" features (Volatility, VIX, Volume), the model learns to avoid trades during high-risk conditions.


## 2. Hypothesis

**The Problem:** Simple technical indicators (like RSI Mean Reversion or Trend Following) are basic and generate signals regardless of the weather of the market. A mean-reversion trade that works in a calm market will perform badly during a volatility spike.

**The Hypothesis:**
Financial markets exhibit distinct regimes (Low Vol/High Vol). A Machine Learning model can learn the non-linear interaction between technical signals and volatility to filter out false positives.

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

**Context Feature:**
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

## 4. Experimentation & Model Evolution

I conducted a comparsion between two algorithms to find the best regime detector.

| Model | Hypothesis | Results |
| :--- | :--- | :--- |
| **Random Forest (Bagging)** | A Random Forest approach will reduce variance and handle noise well. | High stability, but slightly conservative. |
| **XGBoost (Boosting)** | A sequential correction approach will aggressively find edge cases and improve precision. | **Winner.** Better at identifying specific failure modes in high volatility. |

### The Optimisation Process (Grid Search)
I performed a 3-Fold Cross-Validation Grid Search to tune hyperparameters:

* **Random Forest:** Tuned `max_depth` and `min_samples_leaf`. Found that deeper trees (depth=10) with moderate leaf size (10) balanced bias and variance best.
* **XGBoost:** Tuned `learning_rate` and `scale_pos_weight`. The model required a high weight on the minority class (Wins) to overcome the dataset imbalance.

## 5. Results 

### Performance on Unseen Test Data (2022–2024)
The test period covered the 2022 Inflation Bear Market and the 2023 Recovery.

| Strategy | Total Return | Sharpe Ratio | Max Drawdown |
| :--- | :--- | :--- | :--- |
| **S&P 500 (Benchmark)** | +2.08% | 0.15 | -24.46% |
| **Random Forest (Opt)** | -2.72% | 0.01 | -22.86% |
| **XGBoost (Opt)** | **+0.22%** | **0.09** | **-21.93%** |

### Key Findings
1.  **Risk Management Success:** All ML models reduced the Maximum Drawdown compared to the market (-21.9% vs -24.5%). The meta-labeling identified "Crash Regimes" and kept the strategy in cash.
2.  **The Precision-Recall Trade-off:** The Random Forest was too conservative during the 2023 recovery, resulting in a slight loss. XGBoost managed to capture enough upside to stay positive (+0.22%) while still protecting downside.
3.  **Regime Interaction:** Feature Importance analysis confirmed that Volatility and VIX were the primary decision drivers. This supports the hypothesis that market stress is a significant driver of technical strategy failure, often overriding the trade signal itself.



