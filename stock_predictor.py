"""
Stock Price Direction Predictor
================================
A beginner-friendly ML project that predicts whether a stock will go UP or DOWN.

This uses:
- yfinance: to fetch real stock data
- pandas: for data manipulation
- scikit-learn: for the ML model
- matplotlib: for visualization
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def fetch_stock_data(ticker: str, years: int = 2) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
        years: How many years of historical data to fetch
    
    Returns:
        DataFrame with stock price history
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"📈 Fetching {ticker} data from {start_date.date()} to {end_date.date()}...")
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    print(f"✅ Retrieved {len(df)} trading days of data")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw price data.
    
    Features created:
    - Moving averages (5, 10, 20 days)
    - Price momentum (daily returns)
    - Volatility (rolling standard deviation)
    - Volume changes
    - Price relative to moving averages
    """
    df = df.copy()
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()  # Daily percentage change
    df['MA_5'] = df['Close'].rolling(window=5).mean()   # 5-day moving average
    df['MA_10'] = df['Close'].rolling(window=10).mean() # 10-day moving average
    df['MA_20'] = df['Close'].rolling(window=20).mean() # 20-day moving average
    
    # Price relative to moving averages (useful signals!)
    df['Price_vs_MA5'] = df['Close'] / df['MA_5']
    df['Price_vs_MA10'] = df['Close'] / df['MA_10']
    df['Price_vs_MA20'] = df['Close'] / df['MA_20']
    
    # Volatility (how much price swings)
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    
    # Volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Momentum features
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1  # 5-day momentum
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1  # 10-day momentum
    
    # High-Low range (daily volatility)
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Target variable: Will the stock go UP (1) or DOWN (0) tomorrow?
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for ML model.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Select feature columns (exclude raw prices and target)
    feature_columns = [
        'Returns', 'Price_vs_MA5', 'Price_vs_MA10', 'Price_vs_MA20',
        'Volatility', 'Volume_Change', 'Volume_Ratio',
        'Momentum_5', 'Momentum_10', 'Daily_Range'
    ]
    
    # Remove rows with NaN values (from rolling calculations)
    df_clean = df.dropna()
    
    # Remove the last row (no target for it)
    df_clean = df_clean[:-1]
    
    X = df_clean[feature_columns]
    y = df_clean['Target']
    
    # Split: 80% train, 20% test (keeping time order!)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Data split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Random Forest is great for beginners because:
    - Works well out of the box
    - Handles non-linear relationships
    - Shows feature importance
    - Less prone to overfitting
    """
    print("\n🤖 Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Limit tree depth to prevent overfitting
        min_samples_split=10,  # Minimum samples to split a node
        random_state=42        # For reproducibility
    )
    
    model.fit(X_train, y_train)
    print("✅ Model trained!")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_columns):
    """
    Evaluate model performance and show insights.
    """
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Accuracy
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print("\n" + "=" * 50)
    print("📈 MODEL PERFORMANCE")
    print("=" * 50)
    print(f"\nTraining Accuracy: {train_accuracy:.2%}")
    print(f"Testing Accuracy:  {test_accuracy:.2%}")
    print(f"\n(Random guessing would be ~50%)")
    
    # Feature importance
    print("\n🔍 Top Features (what the model finds important):")
    importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for _, row in importance.head(5).iterrows():
        bar = "█" * int(row['Importance'] * 50)
        print(f"   {row['Feature']:20} {bar} {row['Importance']:.3f}")
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, test_pred, target_names=['DOWN', 'UP']))
    
    return test_accuracy, importance


def plot_results(df: pd.DataFrame, model, X_test, y_test, ticker: str):
    """
    Create visualizations of the results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{ticker} Stock Predictor Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Stock price with moving averages
    ax1 = axes[0, 0]
    df_plot = df.tail(100)  # Last 100 days
    ax1.plot(df_plot.index, df_plot['Close'], label='Close Price', linewidth=2)
    ax1.plot(df_plot.index, df_plot['MA_5'], label='5-day MA', alpha=0.7)
    ax1.plot(df_plot.index, df_plot['MA_20'], label='20-day MA', alpha=0.7)
    ax1.set_title('Recent Price History')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Daily returns distribution
    ax2 = axes[0, 1]
    returns = df['Returns'].dropna()
    ax2.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Distribution of Daily Returns')
    ax2.set_xlabel('Daily Return')
    ax2.set_ylabel('Frequency')
    
    # Plot 3: Feature importance
    ax3 = axes[1, 0]
    importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    ax3.barh(importance['Feature'], importance['Importance'], color='teal')
    ax3.set_title('Feature Importance')
    ax3.set_xlabel('Importance')
    
    # Plot 4: Prediction probabilities
    ax4 = axes[1, 1]
    probs = model.predict_proba(X_test)[:, 1]  # Probability of UP
    ax4.hist(probs, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision threshold')
    ax4.set_title('Model Confidence (Probability of UP)')
    ax4.set_xlabel('Probability')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('stock_predictor_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Results saved to 'stock_predictor_results.png'")
    plt.show()


def predict_tomorrow(model, df: pd.DataFrame, feature_columns: list, ticker: str):
    """
    Make a prediction for the next trading day.
    """
    # Get the most recent data point
    latest = df[feature_columns].iloc[-1:]
    
    prediction = model.predict(latest)[0]
    probability = model.predict_proba(latest)[0]
    
    direction = "📈 UP" if prediction == 1 else "📉 DOWN"
    confidence = max(probability) * 100
    
    print("\n" + "=" * 50)
    print(f"🔮 PREDICTION FOR {ticker}")
    print("=" * 50)
    print(f"\nPredicted Direction: {direction}")
    print(f"Model Confidence: {confidence:.1f}%")
    print(f"\n⚠️  Disclaimer: This is for educational purposes only!")
    print("   Never make real investment decisions based on this model.")
    
    return prediction, probability


def main():
    """
    Main function to run the stock predictor.
    """
    print("=" * 50)
    print("🚀 STOCK PRICE DIRECTION PREDICTOR")
    print("=" * 50)
    
    # Configuration
    TICKER = "FIG"  # Try: 'AAPL', 'MSFT', 'TSLA', 'AMZN'
    YEARS = 2        # Years of historical data
    
    # Feature columns used by the model
    feature_columns = [
        'Returns', 'Price_vs_MA5', 'Price_vs_MA10', 'Price_vs_MA20',
        'Volatility', 'Volume_Change', 'Volume_Ratio',
        'Momentum_5', 'Momentum_10', 'Daily_Range'
    ]
    
    # Step 1: Fetch data
    df = fetch_stock_data(TICKER, YEARS)
    
    # Step 2: Create features
    print("\n⚙️  Engineering features...")
    df = create_features(df)
    
    # Step 3: Prepare data for ML
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Step 4: Train the model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate performance
    accuracy, importance = evaluate_model(
        model, X_train, X_test, y_train, y_test, feature_columns
    )
    
    # Step 6: Make prediction for tomorrow
    predict_tomorrow(model, df.dropna(), feature_columns, TICKER)
    
    # Step 7: Visualize results
    plot_results(df, model, X_test, y_test, TICKER)
    
    print("\n✨ Done! Check out 'stock_predictor_results.png' for visualizations.")
    
    return model, df, accuracy


if __name__ == "__main__":
    model, data, accuracy = main()

