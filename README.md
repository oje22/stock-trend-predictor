# 📈 Stock Price Direction Predictor

A machine learning project that predicts whether a stock will go **UP** or **DOWN** the next trading day using historical price data and technical indicators.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project demonstrates the complete machine learning pipeline:
- **Data Collection** → Real-time stock data via Yahoo Finance API
- **Feature Engineering** → Transform raw prices into predictive signals
- **Model Training** → Random Forest classification
- **Evaluation** → Accuracy metrics, feature importance analysis
- **Prediction** → Next-day price direction forecast

## 🧠 Features Engineered

| Feature | Description |
|---------|-------------|
| `Returns` | Daily percentage price change |
| `MA_5, MA_10, MA_20` | Moving averages (trend indicators) |
| `Price_vs_MA` | Price position relative to trend |
| `Volatility` | Rolling standard deviation of returns |
| `Volume_Ratio` | Current vs average trading volume |
| `Momentum` | 5 and 10-day price momentum |
| `Daily_Range` | Intraday price range (High - Low) |

## 📊 Sample Results

### Apple (AAPL)
```
Training Accuracy: 97.92%
Testing Accuracy:  53.61%

Top Features:
  Volume_Ratio    ███████ 0.119
  Daily_Range     ███████ 0.116
  Price_vs_MA20   ██████  0.105
```

### Google (GOOGL)
```
Training Accuracy: 95.31%
Testing Accuracy:  49.48%

Top Features:
  Momentum_5      ███████ 0.113
  Volume_Change   ██████  0.109
  Returns         ██████  0.108
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor
pip install -r requirements.txt
```

### Run
```bash
python stock_predictor.py
```

### Change Stock
Edit line 266 in `stock_predictor.py`:
```python
TICKER = "TSLA"  # Try: 'AAPL', 'GOOGL', 'MSFT', 'AMZN'
```

## 📈 Output Visualization

The model generates `stock_predictor_results.png` containing:
- Price history with moving averages
- Daily returns distribution
- Feature importance chart
- Model confidence distribution

## 🎓 Key Learnings

### 1. Overfitting Detection
The gap between training (~97%) and testing (~50%) accuracy reveals **overfitting** — a critical ML concept where the model memorizes training data but fails to generalize.

### 2. Market Efficiency
~50% accuracy aligns with the **Efficient Market Hypothesis** — stock prices already incorporate available information, making consistent prediction extremely difficult.

### 3. Feature Importance
Volume-based features often outperform price-based ones, suggesting market activity patterns contain predictive signal.

## 🔧 Tech Stack

- **yfinance** — Yahoo Finance API wrapper
- **pandas** — Data manipulation
- **scikit-learn** — Machine learning algorithms
- **matplotlib** — Data visualization
- **NumPy** — Numerical computing

## 📁 Project Structure

```
stock-predictor/
├── stock_predictor.py    # Main ML pipeline
├── requirements.txt      # Dependencies
├── README.md            # Documentation
└── stock_predictor_results.png  # Generated visualization
```

## 🚧 Future Improvements

- [ ] Add technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Implement LSTM neural network for sequence modeling
- [ ] Add backtesting with simulated trading returns
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Cross-validation for more robust evaluation

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock market prediction is inherently uncertain, and this model should not be used for actual trading decisions.

## 📄 License

MIT License — feel free to use and modify for your own projects!

---

*Built as a first machine learning project to learn the fundamentals of data science and predictive modeling.*

